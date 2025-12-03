use crate::error::{Error, Result};
use crate::proto::schema::SparseFloatArray;

/// Represents a single sparse vector row as (index, value) pairs.
/// The indices should be non-negative and less than 2^32-1.
/// Values must not be NaN.
///
/// # Example
/// ```
/// use milvus::SparseVector;
///
/// // Create a sparse vector with 3 non-zero entries
/// let sparse_vec: SparseVector = vec![(5, 0.3), (10, 0.7), (42, 1.2)];
/// ```
pub type SparseVector = Vec<(u32, f32)>;

/// Serializes multiple sparse vectors to protobuf format.
///
/// The binary format for each row is:
/// - Each non-zero entry: 4 bytes (u32 little-endian index) + 4 bytes (f32 little-endian value)
/// - Entries are sorted by index
///
/// # Arguments
/// * `vectors` - Sparse vectors to serialize (takes ownership to avoid cloning)
///
/// # Returns
/// A `SparseFloatArray` protobuf message with:
/// - `contents`: Vec of byte arrays, one per row
/// - `dim`: Maximum dimension across all vectors (max_index + 1)
pub fn sparse_vectors_to_proto(vectors: Vec<SparseVector>) -> SparseFloatArray {
    let mut contents = Vec::with_capacity(vectors.len());
    let mut max_dim = 0i64;

    for mut row in vectors {
        let bytes = sparse_row_to_bytes(&mut row);
        // After sorting, max index is the last element
        if let Some((max_idx, _)) = row.last() {
            max_dim = max_dim.max((*max_idx as i64) + 1);
        }
        contents.push(bytes);
    }

    SparseFloatArray {
        contents,
        dim: max_dim,
    }
}

/// Converts a single sparse vector row to bytes.
///
/// # Format
/// - Entries are sorted by index (ascending)
/// - Each entry: 4 bytes (u32 index, LE) + 4 bytes (f32 value, LE)
///
/// # Arguments
/// * `row` - A sparse vector (index, value) pairs (sorted in place)
///
/// # Returns
/// Byte representation of the sparse vector
pub fn sparse_row_to_bytes(row: &mut SparseVector) -> Vec<u8> {
    // Sort by index to match Milvus format expectations
    row.sort_by_key(|(idx, _)| *idx);

    let mut bytes = Vec::with_capacity(row.len() * 8);
    for (index, value) in row.iter() {
        bytes.extend_from_slice(&index.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Deserializes a single sparse vector row from bytes.
///
/// # Arguments
/// * `bytes` - Byte array containing sparse vector data
///
/// # Returns
/// A `SparseVector` with (index, value) pairs
///
/// # Errors
/// Returns an error if the byte array length is not a multiple of 8
pub fn sparse_row_from_bytes(bytes: &[u8]) -> Result<SparseVector> {
    if bytes.len() % 8 != 0 {
        return Err(Error::SparseVectorError(format!(
            "Sparse vector bytes length must be multiple of 8, got {}",
            bytes.len()
        )));
    }

    let mut result = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let index = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let value = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        result.push((index, value));
    }
    Ok(result)
}

/// Deserializes protobuf format to multiple sparse vectors.
///
/// # Arguments
/// * `proto` - A `SparseFloatArray` protobuf message
///
/// # Returns
/// A vector of `SparseVector` instances
///
/// # Errors
/// Returns an error if any row fails to parse
pub fn sparse_proto_to_vectors(proto: &SparseFloatArray) -> Result<Vec<SparseVector>> {
    proto
        .contents
        .iter()
        .map(|bytes| sparse_row_from_bytes(bytes))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_row_to_bytes() {
        let mut row = vec![(5, 0.5), (10, 1.0), (3, 0.25)];
        let bytes = sparse_row_to_bytes(&mut row);

        // Should be sorted by index: 3, 5, 10
        // Total: 3 entries * 8 bytes = 24 bytes
        assert_eq!(bytes.len(), 24);

        // First entry: index=3, value=0.25
        let idx1 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let val1 = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(idx1, 3);
        assert_eq!(val1, 0.25);

        // Second entry: index=5, value=0.5
        let idx2 = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let val2 = f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        assert_eq!(idx2, 5);
        assert_eq!(val2, 0.5);

        // Third entry: index=10, value=1.0
        let idx3 = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
        let val3 = f32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
        assert_eq!(idx3, 10);
        assert_eq!(val3, 1.0);
    }

    #[test]
    fn test_sparse_row_from_bytes() {
        let mut bytes = Vec::new();
        // Add entry: index=5, value=0.5
        bytes.extend_from_slice(&5u32.to_le_bytes());
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        // Add entry: index=10, value=1.0
        bytes.extend_from_slice(&10u32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());

        let row = sparse_row_from_bytes(&bytes).unwrap();
        assert_eq!(row.len(), 2);
        assert_eq!(row[0], (5, 0.5));
        assert_eq!(row[1], (10, 1.0));
    }

    #[test]
    fn test_sparse_row_from_bytes_invalid_length() {
        let bytes = vec![0u8; 7]; // Not a multiple of 8
        let result = sparse_row_from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_vectors_to_proto() {
        let vectors = vec![
            vec![(5, 0.5), (10, 1.0)],
            vec![(3, 0.25), (15, 1.5)],
            vec![(20, 2.0)],
        ];

        let proto = sparse_vectors_to_proto(vectors);

        // Check number of rows
        assert_eq!(proto.contents.len(), 3);

        // Check dimension (max index + 1 = 20 + 1 = 21)
        assert_eq!(proto.dim, 21);

        // Verify we can parse back
        let parsed = sparse_proto_to_vectors(&proto).unwrap();
        assert_eq!(parsed.len(), 3);
    }

    #[test]
    fn test_sparse_vectors_roundtrip() {
        let original = vec![
            vec![(5, 0.5), (10, 1.0), (42, 2.5)],
            vec![(3, 0.25)],
            vec![(100, 10.0), (200, 20.0)],
        ];

        let proto = sparse_vectors_to_proto(original);
        let parsed = sparse_proto_to_vectors(&proto).unwrap();

        assert_eq!(parsed.len(), 3);

        // First vector (sorted by index)
        assert_eq!(parsed[0], vec![(5, 0.5), (10, 1.0), (42, 2.5)]);

        // Second vector
        assert_eq!(parsed[1], vec![(3, 0.25)]);

        // Third vector (sorted)
        assert_eq!(parsed[2], vec![(100, 10.0), (200, 20.0)]);
    }

    #[test]
    fn test_empty_sparse_vector() {
        let vectors = vec![vec![], vec![(5, 0.5)]];
        let proto = sparse_vectors_to_proto(vectors);

        assert_eq!(proto.contents.len(), 2);
        assert_eq!(proto.contents[0].len(), 0); // Empty vector
        assert_eq!(proto.dim, 6); // Max index 5 + 1

        let parsed = sparse_proto_to_vectors(&proto).unwrap();
        assert_eq!(parsed[0].len(), 0);
        assert_eq!(parsed[1], vec![(5, 0.5)]);
    }
}
