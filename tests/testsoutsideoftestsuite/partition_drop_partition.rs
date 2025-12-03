#[path = "../common/mod.rs"]
mod common;

use common::*;
use milvus::client::*;

#[tokio::test]
async fn test_drop_partition() {
    let (client, collection) = create_test_collection(true).await.unwrap();
    client
        .create_partition(collection.name().to_string(), "test_partition".to_string())
        .await
        .unwrap();

    // Release the partition before dropping it
    let release_result = client
        .release_partitions(
            collection.name().to_string(),
            vec!["test_partition".to_string()],
        )
        .await;
    assert!(
        release_result.is_ok(),
        "Failed to release partition: {:?}",
        release_result
    );

    let result = client
        .drop_partition(collection.name().to_string(), "test_partition".to_string())
        .await;

    assert!(result.is_ok());
}
