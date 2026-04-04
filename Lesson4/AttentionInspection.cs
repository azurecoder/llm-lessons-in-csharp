namespace ukazure.llm.Lesson4;

internal sealed class AttentionInspection
{
    public required double[] Weights { get; init; }
    public required double[] Probabilities { get; init; }
}
