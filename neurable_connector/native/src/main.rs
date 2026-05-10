//! Headless MW75 CSV streamer. Opens BLE + activation + RFCOMM and dumps every
//! EEG packet to stdout as `ts_us,counter,ch1,ch2,...,ch12` (no header).
//! Logs go to stderr at WARN by default; override with `RUST_LOG=info`.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use log::{info, warn};
use mw75::mw75_client::{Mw75Client, Mw75ClientConfig};
use mw75::types::Mw75Event;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let config = Mw75ClientConfig {
        scan_timeout_secs: 10,
        name_pattern: "MW75".into(),
        ..Default::default()
    };
    let client = Mw75Client::new(config);

    let (mut rx, handle) = client.connect().await?;
    let handle = Arc::new(handle);
    handle.start().await?;
    info!("Activation complete.");

    #[cfg(feature = "rfcomm")]
    let _rfcomm = {
        let bt_address = handle.peripheral_id();
        info!("Disconnecting BLE before RFCOMM…");
        handle.disconnect_ble().await.ok();
        let rfcomm_handle = handle.clone();
        match mw75::rfcomm::start_rfcomm_stream(rfcomm_handle, &bt_address).await {
            Ok(rfcomm) => {
                info!("RFCOMM reader task started");
                Some(rfcomm)
            }
            Err(e) => {
                warn!("RFCOMM failed: {e}");
                None
            }
        }
    };

    info!("entering event loop…");
    while let Some(event) = rx.recv().await {
        match event {
            Mw75Event::Eeg(pkt) => {
                let ts_us = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_micros())
                    .unwrap_or(0);
                let mut line = format!("{ts_us},{}", pkt.counter);
                for v in &pkt.channels {
                    line.push_str(&format!(",{:.4}", v));
                }
                println!("{line}");
            }
            Mw75Event::Battery(b) => info!("Battery: {}%", b.level),
            Mw75Event::Activated(s) => info!(
                "Activated: EEG={}, Raw={}",
                s.eeg_enabled, s.raw_mode_enabled
            ),
            Mw75Event::Disconnected => {
                warn!("Disconnected — exiting");
                break;
            }
            Mw75Event::Connected(name) => info!("Connected: {name}"),
            Mw75Event::RawData(data) => info!("[RAW] {} bytes", data.len()),
            Mw75Event::OtherEvent { event_id, counter, raw } => info!(
                "[OTHER] id={event_id} cnt={counter} len={}",
                raw.len()
            ),
        }
    }
    Ok(())
}
