use std::io::{self, Read, Write};

use rkyv::{
	Archive, Deserialize, Serialize,
	api::high::{HighSerializer, HighValidator},
	bytecheck::CheckBytes,
	de::Pool,
	rancor::{self, Strategy},
	ser::allocator::ArenaHandle,
	util::AlignedVec,
};

pub trait ReadFrom: Archive + Sized {
	fn read_from<R: Read>(reader: &mut R) -> io::Result<Self>;
}

impl<T> ReadFrom for T
where
	T: Archive,
	T::Archived: for<'a> CheckBytes<HighValidator<'a, rancor::Error>> + Deserialize<T, Strategy<Pool, rancor::Error>>,
{
	fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
		let mut len = [0u8; 2];
		reader.read_exact(&mut len)?;
		let len = u16::from_le_bytes(len) as usize;
		let mut buf = vec![0u8; len];
		reader.read_exact(&mut buf)?;
		let archived = rkyv::access::<Self::Archived, rancor::Error>(&buf)
			.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
		rkyv::deserialize(archived).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
	}
}

pub trait WriteTo {
	fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()>;
}

impl<T> WriteTo for T
where
	T: for<'a> Serialize<HighSerializer<AlignedVec, ArenaHandle<'a>, rancor::Error>>,
{
	fn write_to<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
		let bytes = rkyv::to_bytes::<rancor::Error>(self).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
		let len = bytes.len() as u16;
		writer.write_all(&len.to_le_bytes())?;
		writer.write_all(&bytes)?;
		Ok(())
	}
}
