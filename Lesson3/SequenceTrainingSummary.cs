namespace ukazure.llm.Lesson3;

internal sealed class SequenceTrainingSummary
{
    public required double InitialLoss { get; init; }
    public required double FinalLoss { get; init; }
    public required List<(int Epoch, double Loss)> Checkpoints { get; init; }
}
