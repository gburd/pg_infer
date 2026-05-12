//! Custom `tracing` layer that routes log events to PostgreSQL's `elog()`
//! system via pgrx logging macros.
//!
//! Maps tracing levels to PG log levels:
//! - ERROR → pgrx::warning!() (PG ERROR would abort the transaction)
//! - WARN  → pgrx::warning!()
//! - INFO  → pgrx::info!()
//! - DEBUG → pgrx::debug1!()
//! - TRACE → pgrx::debug2!()

use tracing::field::{Field, Visit};
use tracing::Level;
use tracing_subscriber::layer::Context;
use tracing_subscriber::Layer;

/// A tracing subscriber layer that forwards events to PostgreSQL's logging
/// infrastructure via pgrx.
pub struct PgLogLayer;

impl<S> Layer<S> for PgLogLayer
where
    S: tracing::Subscriber,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = PgLogVisitor {
            message: String::new(),
        };
        event.record(&mut visitor);

        let target = event.metadata().target();
        let msg = if visitor.message.is_empty() {
            format!("[{}] (no message)", target)
        } else {
            format!("[{}] {}", target, visitor.message)
        };

        match *event.metadata().level() {
            Level::ERROR | Level::WARN => {
                pgrx::warning!("{}", msg);
            }
            Level::INFO => {
                pgrx::info!("{}", msg);
            }
            Level::DEBUG => {
                pgrx::debug1!("{}", msg);
            }
            Level::TRACE => {
                pgrx::debug2!("{}", msg);
            }
        }
    }
}

/// Visitor that extracts the `message` field from tracing events.
struct PgLogVisitor {
    message: String,
}

impl Visit for PgLogVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{:?}", value);
        } else if self.message.is_empty() {
            self.message = format!("{}={:?}", field.name(), value);
        } else {
            self.message
                .push_str(&format!(" {}={:?}", field.name(), value));
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else if self.message.is_empty() {
            self.message = format!("{}={}", field.name(), value);
        } else {
            self.message
                .push_str(&format!(" {}={}", field.name(), value));
        }
    }
}
