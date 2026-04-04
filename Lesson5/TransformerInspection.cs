namespace ukazure.llm.Lesson5;

internal sealed class TransformerInspection
{
    public required double[] AttentionWeights { get; init; }
    public required double[] Probabilities { get; init; }
}
