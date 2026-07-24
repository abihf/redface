use serde::Serialize;
use serde::de::DeserializeOwned;
use std::io::{Read, Write};

mod config;
mod descriptor;
mod protocol;
mod simd;

pub use config::*;
pub use descriptor::*;
pub use protocol::*;

/// Read a JSON value from any reader. Implemented automatically for every
/// type that derives [`Deserialize`](serde::Deserialize).
pub trait ReadJson: Sized {
	fn read_json(reader: impl Read) -> serde_json::Result<Self>;
}

impl<T: DeserializeOwned> ReadJson for T {
	fn read_json(reader: impl Read) -> serde_json::Result<Self> {
		let mut deserializer = serde_json::Deserializer::from_reader(reader);
		T::deserialize(&mut deserializer)
	}
}

pub trait WriteJson {
	fn write_json(&self, writer: impl Write) -> serde_json::Result<()>;
}

impl<T: Serialize> WriteJson for T {
	fn write_json(&self, writer: impl Write) -> serde_json::Result<()> {
		serde_json::to_writer(writer, self)
	}
}
