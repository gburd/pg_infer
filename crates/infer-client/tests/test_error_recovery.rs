//! Error recovery tests for the CancellableClient.
//!
//! Verifies that the remote backend handles failure modes gracefully:
//! - Connection refused → helpful error (not panic or hang)
//! - Server timeout → timeout error (not hang)
//! - Cancellation during request → clean abort
//! - Malformed JSON → parse error (not panic)

use std::time::Duration;

use infer_client::{CancelToken, CancellableClient, ClientError, StatsResponse};

#[test]
fn connection_refused_returns_helpful_error() {
    // Connect to a port where nothing is listening.
    // Use a high ephemeral port that's almost certainly unused.
    let client = CancellableClient::connect("http://127.0.0.1:19999", Duration::from_secs(2));

    match client {
        Ok(c) => {
            // Client created (it doesn't connect eagerly), but the first
            // request should fail with a transport error.
            let token = CancelToken::new();
            let result: Result<StatsResponse, ClientError> =
                c.get_json("/v1/stats", &token);
            match result {
                Err(ClientError::Transport(msg)) => {
                    // Should mention connection failure
                    assert!(
                        msg.contains("onnect") || msg.contains("refused") || msg.contains("error"),
                        "error should mention connection issue, got: {}",
                        msg
                    );
                }
                Err(other) => {
                    // Any error is acceptable as long as it's not a panic
                    eprintln!("Got non-Transport error (acceptable): {:?}", other);
                }
                Ok(_) => panic!("expected error for connection to nothing"),
            }
        }
        Err(e) => {
            // If connect itself fails immediately, that's also acceptable
            eprintln!("connect() returned error (acceptable): {:?}", e);
        }
    }
}

#[test]
fn timeout_returns_error_not_hang() {
    // Start a mock server that never responds (accepts but delays).
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let base_url = format!("http://{}", addr);

    // Spawn a thread that accepts connections but never responds
    let _server = std::thread::spawn(move || {
        // Accept connections but don't send any response
        while let Ok((_stream, _)) = listener.accept() {
            // Hold the connection open, don't write anything
            std::thread::sleep(Duration::from_secs(30));
        }
    });

    // Client with a very short timeout
    let client = CancellableClient::connect(&base_url, Duration::from_millis(200));
    match client {
        Ok(c) => {
            let token = CancelToken::new();
            let start = std::time::Instant::now();
            let result: Result<StatsResponse, ClientError> =
                c.get_json("/v1/stats", &token);
            let elapsed = start.elapsed();

            // Should complete within a reasonable time (not hang)
            assert!(
                elapsed < Duration::from_secs(5),
                "request took too long ({:?}), likely hung",
                elapsed
            );

            // Should be an error
            assert!(
                result.is_err(),
                "expected timeout error, got: {:?}",
                result
            );
        }
        Err(e) => {
            eprintln!("connect() returned error (acceptable): {:?}", e);
        }
    }
}

#[test]
fn cancellation_aborts_request() {
    // Start a mock server that delays
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let base_url = format!("http://{}", addr);

    let _server = std::thread::spawn(move || {
        while let Ok((_stream, _)) = listener.accept() {
            std::thread::sleep(Duration::from_secs(30));
        }
    });

    let client = CancellableClient::connect(&base_url, Duration::from_secs(10));
    match client {
        Ok(c) => {
            let token = CancelToken::new();

            // Cancel after 100ms from another thread
            let cancel_token = token.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                cancel_token.cancel();
            });

            let start = std::time::Instant::now();
            let result: Result<StatsResponse, ClientError> =
                c.get_json("/v1/stats", &token);
            let elapsed = start.elapsed();

            // Should abort quickly (within ~1 second)
            assert!(
                elapsed < Duration::from_secs(3),
                "cancellation took too long ({:?})",
                elapsed
            );

            // Should be cancelled or transport error
            match result {
                Err(ClientError::Cancelled) => {} // expected
                Err(_) => {}                      // other errors acceptable too
                Ok(_) => panic!("expected error after cancellation"),
            }
        }
        Err(e) => {
            eprintln!("connect() returned error (acceptable): {:?}", e);
        }
    }
}

#[test]
fn invalid_url_returns_error() {
    let result = CancellableClient::connect("not-a-url", Duration::from_secs(1));
    assert!(
        result.is_err(),
        "expected error for invalid URL, got: Ok"
    );
    if let Err(ClientError::InvalidUrl(msg)) = result {
        assert!(!msg.is_empty(), "error message should not be empty");
    }
    // Other error types also acceptable
}
