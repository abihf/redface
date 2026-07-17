use std::fmt;
use std::io::{self, BufRead, Write};
use std::num::ParseIntError;
use std::str::FromStr;

mod simd;

pub const DESCRIPTOR_LEN: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Descriptor(pub [f32; DESCRIPTOR_LEN]);

#[derive(Debug)]
pub enum DescriptorError {
	WrongValueCount { found: usize },
	InvalidFloat { token: String },
	Io(io::Error),
}

impl fmt::Display for DescriptorError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::WrongValueCount { found } => {
				write!(f, "expected {DESCRIPTOR_LEN} descriptor values, found {found}")
			}
			Self::InvalidFloat { token } => write!(f, "invalid float value: {token}"),
			Self::Io(err) => err.fmt(f),
		}
	}
}

impl std::error::Error for DescriptorError {}

impl From<io::Error> for DescriptorError {
	fn from(value: io::Error) -> Self {
		Self::Io(value)
	}
}

impl Descriptor {
	/// Cosine similarity in `[-1, 1]`; higher means more similar.
	/// Returns 0.0 when either vector has zero magnitude.
	pub fn cosine_similarity(&self, other: &Self) -> f64 {
		let (dot, norm_self, norm_other) = simd::dot_and_norms(&self.0, &other.0);
		let denom = norm_self.sqrt() * norm_other.sqrt();
		if denom == 0.0 { 0.0 } else { dot / denom }
	}

	pub fn middle(&self, other: &Self) -> Self {
		let mut result = [0.0; DESCRIPTOR_LEN];
		for (index, slot) in result.iter_mut().enumerate() {
			*slot = (self.0[index] + other.0[index]) / 2.0;
		}
		Self(result)
	}

	pub fn parse_line(line: &str) -> Result<Self, DescriptorError> {
		let tokens: Vec<_> = line.split_whitespace().collect();
		if tokens.len() != DESCRIPTOR_LEN {
			return Err(DescriptorError::WrongValueCount { found: tokens.len() });
		}

		let mut values = [0.0; DESCRIPTOR_LEN];
		for (index, token) in tokens.iter().enumerate() {
			values[index] = parse_go_float(token)?;
		}

		Ok(Self(values))
	}

	pub fn encode_line(&self) -> String {
		self.0
			.iter()
			.map(|value| format_go_float(*value))
			.collect::<Vec<_>>()
			.join(" ")
	}

	pub fn write_line(&self, mut writer: impl Write) -> io::Result<()> {
		writer.write_all(self.encode_line().as_bytes())
	}
}

pub fn read_descriptors(reader: impl BufRead) -> Result<Vec<Descriptor>, DescriptorError> {
	let mut descriptors = Vec::new();
	for line in reader.lines() {
		let line = line?;
		let trimmed = line.trim();
		if trimmed.is_empty() {
			continue;
		}
		descriptors.push(Descriptor::parse_line(trimmed)?);
	}
	Ok(descriptors)
}

pub fn write_descriptors(mut writer: impl Write, descriptors: &[Descriptor]) -> Result<(), DescriptorError> {
	for descriptor in descriptors {
		descriptor.write_line(&mut writer)?;
		writer.write_all(b"\n")?;
	}
	Ok(())
}

fn parse_go_float(token: &str) -> Result<f32, DescriptorError> {
	if looks_like_hex_float(token) {
		parse_hex_float(token).map_err(|_| DescriptorError::InvalidFloat {
			token: token.to_owned(),
		})
	} else {
		f32::from_str(token).map_err(|_| DescriptorError::InvalidFloat {
			token: token.to_owned(),
		})
	}
}

fn looks_like_hex_float(token: &str) -> bool {
	let unsigned = token.strip_prefix(['+', '-']).unwrap_or(token);
	unsigned.starts_with("0x") && unsigned.contains(['p', 'P'])
}

fn parse_hex_float(token: &str) -> Result<f32, ParseIntError> {
	let (negative, unsigned) = match token.as_bytes().first() {
		Some(b'-') => (true, &token[1..]),
		Some(b'+') => (false, &token[1..]),
		_ => (false, token),
	};

	let unsigned = unsigned.strip_prefix("0x").unwrap_or(unsigned);
	let (mantissa, exponent) = unsigned
		.split_once(['p', 'P'])
		.expect("hex float tokens are validated before parsing");

	let exponent = exponent.parse::<i32>().unwrap_or(0);
	let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));

	let whole_value = if whole.is_empty() {
		0.0
	} else {
		u64::from_str_radix(whole, 16)? as f64
	};

	let mut fraction_value = 0.0_f64;
	let mut place = 1.0_f64 / 16.0_f64;
	for ch in fraction.bytes() {
		let digit = hex_digit(ch) as f64;
		fraction_value += digit * place;
		place /= 16.0;
	}

	let value = (whole_value + fraction_value) * 2_f64.powi(exponent);
	Ok(if negative { -(value as f32) } else { value as f32 })
}

fn hex_digit(ch: u8) -> u8 {
	match ch {
		b'0'..=b'9' => ch - b'0',
		b'a'..=b'f' => 10 + (ch - b'a'),
		b'A'..=b'F' => 10 + (ch - b'A'),
		_ => 0,
	}
}

fn format_go_float(value: f32) -> String {
	if value.is_nan() {
		return "NaN".to_owned();
	}
	if value.is_infinite() {
		return if value.is_sign_negative() {
			"-Inf".to_owned()
		} else {
			"+Inf".to_owned()
		};
	}
	if value == 0.0 {
		return if value.is_sign_negative() {
			"-0x0p+00".to_owned()
		} else {
			"0x0p+00".to_owned()
		};
	}

	let abs = value.abs();
	let bits = abs.to_bits();
	let exponent_bits = ((bits >> 23) & 0xff) as i32;
	let fraction_bits = bits & 0x7f_ffff;

	let (exponent, numerator, denominator_bits) = if exponent_bits == 0 {
		let highest = 31 - fraction_bits.leading_zeros() as i32;
		let denominator_bits = highest;
		let numerator = u64::from(fraction_bits - (1_u32 << highest));
		(highest - 149, numerator, denominator_bits)
	} else {
		(exponent_bits - 127, u64::from(fraction_bits), 23)
	};

	let fractional = format_fractional_hex(numerator, denominator_bits as u32);
	let sign = if value.is_sign_negative() { "-" } else { "" };
	if fractional.is_empty() {
		format!("{sign}0x1p{exponent:+03}")
	} else {
		format!("{sign}0x1.{fractional}p{exponent:+03}")
	}
}

fn format_fractional_hex(mut numerator: u64, denominator_bits: u32) -> String {
	if numerator == 0 || denominator_bits == 0 {
		return String::new();
	}

	let denominator = 1_u64 << denominator_bits;
	let digits = denominator_bits.div_ceil(4) as usize;
	let mut out = String::with_capacity(digits);
	for _ in 0..digits {
		numerator *= 16;
		let digit = (numerator / denominator) as u8;
		numerator %= denominator;
		out.push(char::from_digit(u32::from(digit), 16).expect("valid hex digit"));
		if numerator == 0 {
			break;
		}
	}

	while out.ends_with('0') {
		out.pop();
	}
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn middle_averages_all_components() {
		let left = Descriptor([2.0; DESCRIPTOR_LEN]);
		let right = Descriptor([4.0; DESCRIPTOR_LEN]);

		let result = left.middle(&right);

		assert_eq!(result, Descriptor([3.0; DESCRIPTOR_LEN]));
	}

	#[test]
	fn cosine_similarity_is_one_for_identical_vectors() {
		let value = Descriptor([1.5; DESCRIPTOR_LEN]);

		assert!((value.cosine_similarity(&value) - 1.0).abs() < 1e-12);
	}

	#[test]
	fn cosine_similarity_is_zero_for_orthogonal_vectors() {
		let mut left = [0.0; DESCRIPTOR_LEN];
		let mut right = [0.0; DESCRIPTOR_LEN];
		left[0] = 1.0;
		right[1] = 1.0;

		let similarity = Descriptor(left).cosine_similarity(&Descriptor(right));

		assert!(similarity.abs() < 1e-12);
	}

	#[test]
	fn cosine_similarity_ignores_magnitude() {
		let mut left = [0.0; DESCRIPTOR_LEN];
		let mut right = [0.0; DESCRIPTOR_LEN];
		left[3] = 1.0;
		right[3] = 7.0;

		let similarity = Descriptor(left).cosine_similarity(&Descriptor(right));

		assert!((similarity - 1.0).abs() < 1e-12);
	}

	#[test]
	fn parse_and_encode_hex_float_round_trip() {
		let value = parse_go_float("-0x1.e8b1e6p-04").expect("hex float parses");

		assert_eq!(format_go_float(value), "-0x1.e8b1e6p-04");
	}
}
