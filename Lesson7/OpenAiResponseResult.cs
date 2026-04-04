namespace ukazure.llm.Lesson7;

internal sealed class OpenAiResponseResult
{
    public required bool UsedLiveModel { get; init; }
    public required string Explanation { get; init; }
    public required string Answer { get; init; }
}
