namespace ukazure.llm.Lesson5;

internal sealed class TransformerTrainingSummary
{
    public required double InitialLoss { get; init; }
    public required double FinalLoss { get; init; }
    public required List<(int Epoch, double Loss)> Checkpoints { get; init; }
}
