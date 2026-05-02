namespace ukazure.llm.NanoGpt;

internal sealed record NanoGptConfig(
    int VocabularySize,
    int BlockSize,
    int LayerCount,
    int HeadCount,
    int EmbeddingSize,
    int FeedForwardSize);
