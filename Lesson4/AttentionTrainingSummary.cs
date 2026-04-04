namespace ukazure.llm.Lesson4;

internal sealed class AttentionTrainingSummary
{
    public required double InitialLoss { get; init; }
    public required double FinalLoss { get; init; }
    public required List<(int Epoch, double Loss)> Checkpoints { get; init; }
}
