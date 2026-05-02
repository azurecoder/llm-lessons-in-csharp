namespace ukazure.llm.NanoGpt;

internal sealed class NanoGptRun
{
    public required string TrainingDocumentPath { get; init; }
    public required string TrainingDocument { get; init; }
    public required string TrainingDocumentPreview { get; init; }
    public required CharacterTokenizer Tokenizer { get; init; }
    public required NanoGptConfig Config { get; init; }
    public required IReadOnlyList<(int Step, double Loss)> Checkpoints { get; init; }
    public required string Prompt { get; init; }
    public required string Sample { get; init; }
}
