//! Persistent on-disk residual cache for template-fixed layers.
//!
//! Format: header + entry table + data section (all little-endian).
//!
//! Header (20 bytes):
//!   magic: [u8; 4] = b"RCCH"
//!   version: u32 = 1
//!   num_entries: u32
//!   hidden_size: u32
//!   num_layers: u32
//!
//! Entry table (24 bytes each):
//!   template_hash: u64
//!   layer: u32
//!   data_offset: u64
//!   data_len: u32  (in bytes, = seq_len * hidden_size * 4)
//!
//! Data section:
//!   Raw f32 arrays, packed sequentially.

use std::path::Path;

use memmap2::Mmap;

use crate::error::VindexError;

const MAGIC: [u8; 4] = *b"RCCH";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 20;
const ENTRY_SIZE: usize = 24;

/// A single cached residual entry descriptor.
#[derive(Debug, Clone)]
pub struct ResidualEntry {
    pub template_hash: u64,
    pub layer: usize,
    pub offset: usize,
    pub length: usize,
}

/// Read-only mmap-backed residual cache.
pub struct ResidualCache {
    mmap: Mmap,
    entries: Vec<ResidualEntry>,
    pub hidden_size: usize,
}

impl ResidualCache {
    /// Open an existing residual cache file.
    pub fn open(path: &Path) -> Result<Self, VindexError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;

        if mmap.len() < HEADER_SIZE {
            return Err(VindexError::Parse("residual cache too small".into()));
        }
        let data = &mmap[..];

        // Parse header
        if data[0..4] != MAGIC {
            return Err(VindexError::Parse("bad residual cache magic".into()));
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(VindexError::Parse(format!(
                "unsupported residual cache version {version}"
            )));
        }
        let num_entries = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let hidden_size = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let _num_layers = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;

        // Parse entry table
        let table_start = HEADER_SIZE;
        let table_end = table_start + num_entries * ENTRY_SIZE;
        if mmap.len() < table_end {
            return Err(VindexError::Parse("residual cache truncated".into()));
        }

        let mut entries = Vec::with_capacity(num_entries);
        for i in 0..num_entries {
            let base = table_start + i * ENTRY_SIZE;
            let template_hash = u64::from_le_bytes(data[base..base + 8].try_into().unwrap());
            let layer = u32::from_le_bytes(data[base + 8..base + 12].try_into().unwrap()) as usize;
            let data_offset =
                u64::from_le_bytes(data[base + 12..base + 20].try_into().unwrap()) as usize;
            let data_len =
                u32::from_le_bytes(data[base + 20..base + 24].try_into().unwrap()) as usize;
            entries.push(ResidualEntry {
                template_hash,
                layer,
                offset: data_offset,
                length: data_len,
            });
        }

        Ok(Self {
            mmap,
            entries,
            hidden_size,
        })
    }

    /// Look up a cached residual by template hash and layer.
    /// Returns the raw f32 slice if found.
    pub fn get(&self, template_hash: u64, layer: usize) -> Option<&[f32]> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.template_hash == template_hash && e.layer == layer)?;
        let start = entry.offset;
        let end = start + entry.length;
        if end > self.mmap.len() {
            return None;
        }
        let bytes = &self.mmap[start..end];
        // Safety: data section written by ResidualCacheBuilder as contiguous f32 values.
        // The pointer alignment is guaranteed by the sequential write layout (data_offset
        // is always HEADER + table size, both multiples of 4, and each entry length is a
        // multiple of 4).
        let floats =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, entry.length / 4) };
        Some(floats)
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Builder for writing residual cache files.
pub struct ResidualCacheBuilder {
    hidden_size: usize,
    num_layers: usize,
    entries: Vec<(u64, usize, Vec<f32>)>, // (template_hash, layer, residual_data)
}

impl ResidualCacheBuilder {
    /// Create a new builder.
    pub fn new(hidden_size: usize, num_layers: usize) -> Self {
        Self {
            hidden_size,
            num_layers,
            entries: Vec::new(),
        }
    }

    /// Add a residual for a (template_hash, layer) pair.
    pub fn add(&mut self, template_hash: u64, layer: usize, residual: &[f32]) {
        self.entries.push((template_hash, layer, residual.to_vec()));
    }

    /// Write the cache to disk.
    pub fn write(&self, path: &Path) -> Result<(), VindexError> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Header
        file.write_all(&MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&(self.entries.len() as u32).to_le_bytes())?;
        file.write_all(&(self.hidden_size as u32).to_le_bytes())?;
        file.write_all(&(self.num_layers as u32).to_le_bytes())?;

        // Compute data offsets
        let table_end = HEADER_SIZE + self.entries.len() * ENTRY_SIZE;
        let mut data_offset = table_end;

        // Entry table
        for (template_hash, layer, data) in &self.entries {
            let data_len = data.len() * 4; // f32 = 4 bytes
            file.write_all(&template_hash.to_le_bytes())?;
            file.write_all(&(*layer as u32).to_le_bytes())?;
            file.write_all(&(data_offset as u64).to_le_bytes())?;
            file.write_all(&(data_len as u32).to_le_bytes())?;
            data_offset += data_len;
        }

        // Data section
        for (_, _, data) in &self.entries {
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            file.write_all(bytes)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_residual_cache.bin");

        let hidden = 64;
        let num_layers = 4;
        let seq_len = 3;

        // Build
        let mut builder = ResidualCacheBuilder::new(hidden, num_layers);
        let data1: Vec<f32> = (0..seq_len * hidden).map(|i| i as f32 * 0.1).collect();
        let data2: Vec<f32> = (0..seq_len * hidden).map(|i| i as f32 * 0.2).collect();
        builder.add(0xDEAD, 0, &data1);
        builder.add(0xDEAD, 1, &data2);
        builder.add(0xBEEF, 0, &data1);
        builder.write(&path).unwrap();

        // Read
        let cache = ResidualCache::open(&path).unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.hidden_size, hidden);

        // Lookup
        let got = cache.get(0xDEAD, 0).unwrap();
        assert_eq!(got.len(), seq_len * hidden);
        assert!((got[0] - 0.0).abs() < 1e-6);
        assert!((got[1] - 0.1).abs() < 1e-6);

        let got2 = cache.get(0xDEAD, 1).unwrap();
        assert!((got2[1] - 0.2).abs() < 1e-6);

        // Missing
        assert!(cache.get(0xDEAD, 5).is_none());
        assert!(cache.get(0xCAFE, 0).is_none());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }
}
