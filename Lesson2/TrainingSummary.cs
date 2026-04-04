namespace ukazure.llm.Lesson2;

internal sealed class TrainingSummary
{
    public required double InitialLoss { get; init; }
    public required double FinalLoss { get; init; }
    public required List<(int Epoch, double Loss)> Checkpoints { get; init; }
}
