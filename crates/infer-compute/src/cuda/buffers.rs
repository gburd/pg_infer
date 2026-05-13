//! GPU buffer pool and host<->device transfer utilities.
//!
//! Manages device memory allocation with size-bucketed reuse to minimize
//! cudaMalloc overhead in hot loops. Buffers are returned to the pool on drop
//! and reused for subsequent allocations of the same size class.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaDevice, CudaSlice};

/// Size bucket: round up to next power of two for reuse efficiency.
fn bucket_size(bytes: usize) -> usize {
    if bytes == 0 {
        return 0;
    }
    bytes.next_power_of_two()
}

/// A device buffer wrapper that returns to the pool on drop.
pub struct GpuBuffer {
    /// The underlying device allocation (f32 elements).
    pub slice: CudaSlice<f32>,
    /// Actual number of f32 elements requested (may be less than allocation).
    pub len: usize,
}

/// A device buffer wrapper for raw bytes.
pub struct GpuBufferBytes {
    /// The underlying device allocation (u8 elements).
    pub slice: CudaSlice<u8>,
    /// Actual number of bytes requested.
    pub len: usize,
}

/// Pool of reusable GPU buffers, bucketed by allocation size.
///
/// Avoids repeated cudaMalloc/cudaFree in the inference hot path.
/// Thread-safe via internal mutex.
pub struct BufferPool {
    device: Arc<CudaDevice>,
    /// Free f32 buffers indexed by bucket size (in f32 elements).
    f32_pool: Mutex<HashMap<usize, Vec<CudaSlice<f32>>>>,
    /// Free u8 buffers indexed by bucket size (in bytes).
    u8_pool: Mutex<HashMap<usize, Vec<CudaSlice<u8>>>>,
}

impl BufferPool {
    /// Create a new buffer pool for the given device.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            f32_pool: Mutex::new(HashMap::new()),
            u8_pool: Mutex::new(HashMap::new()),
        }
    }

    /// Allocate (or reuse) a device buffer for `len` f32 elements.
    pub fn alloc_f32(&self, len: usize) -> Result<GpuBuffer, super::CudaError> {
        let bucket = bucket_size(len);
        let mut pool = self.f32_pool.lock().unwrap();

        let slice = if let Some(free_list) = pool.get_mut(&bucket) {
            if let Some(buf) = free_list.pop() {
                buf
            } else {
                self.device
                    .alloc_zeros::<f32>(bucket)
                    .map_err(|e| super::CudaError::MemAlloc(e.to_string()))?
            }
        } else {
            self.device
                .alloc_zeros::<f32>(bucket)
                .map_err(|e| super::CudaError::MemAlloc(e.to_string()))?
        };

        Ok(GpuBuffer { slice, len })
    }

    /// Allocate (or reuse) a device buffer for `len` bytes.
    pub fn alloc_bytes(&self, len: usize) -> Result<GpuBufferBytes, super::CudaError> {
        let bucket = bucket_size(len);
        let mut pool = self.u8_pool.lock().unwrap();

        let slice = if let Some(free_list) = pool.get_mut(&bucket) {
            if let Some(buf) = free_list.pop() {
                buf
            } else {
                self.device
                    .alloc_zeros::<u8>(bucket)
                    .map_err(|e| super::CudaError::MemAlloc(e.to_string()))?
            }
        } else {
            self.device
                .alloc_zeros::<u8>(bucket)
                .map_err(|e| super::CudaError::MemAlloc(e.to_string()))?
        };

        Ok(GpuBufferBytes { slice, len })
    }

    /// Return an f32 buffer to the pool for reuse.
    pub fn return_f32(&self, buf: GpuBuffer) {
        let bucket = bucket_size(buf.len);
        let mut pool = self.f32_pool.lock().unwrap();
        pool.entry(bucket).or_default().push(buf.slice);
    }

    /// Return a byte buffer to the pool for reuse.
    pub fn return_bytes(&self, buf: GpuBufferBytes) {
        let bucket = bucket_size(buf.len);
        let mut pool = self.u8_pool.lock().unwrap();
        pool.entry(bucket).or_default().push(buf.slice);
    }

    /// Upload f32 data from host to a pooled device buffer.
    pub fn upload_f32(&self, data: &[f32]) -> Result<GpuBuffer, super::CudaError> {
        let len = data.len();
        let slice = self.device
            .htod_sync_copy(data)
            .map_err(|e| super::CudaError::Transfer(e.to_string()))?;
        Ok(GpuBuffer { slice, len })
    }

    /// Upload raw bytes from host to a pooled device buffer.
    pub fn upload_bytes(&self, data: &[u8]) -> Result<GpuBufferBytes, super::CudaError> {
        let len = data.len();
        let slice = self.device
            .htod_sync_copy(data)
            .map_err(|e| super::CudaError::Transfer(e.to_string()))?;
        Ok(GpuBufferBytes { slice, len })
    }

    /// Download f32 data from device to host.
    pub fn download_f32(&self, buf: &GpuBuffer) -> Result<Vec<f32>, super::CudaError> {
        let data = self.device
            .dtoh_sync_copy(&buf.slice)
            .map_err(|e| super::CudaError::Transfer(e.to_string()))?;
        // Trim to actual length (buffer may be over-allocated due to bucketing).
        Ok(data[..buf.len].to_vec())
    }

    /// Get the underlying device reference.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Total number of cached buffers across all pools.
    pub fn cached_count(&self) -> usize {
        let f32_count: usize = self.f32_pool.lock().unwrap().values().map(|v| v.len()).sum();
        let u8_count: usize = self.u8_pool.lock().unwrap().values().map(|v| v.len()).sum();
        f32_count + u8_count
    }
}
