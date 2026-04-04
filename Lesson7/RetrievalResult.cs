namespace ukazure.llm.Lesson7;

internal sealed class RetrievalResult
{
    public required KnowledgeDocument Document { get; init; }
    public required int Score { get; init; }
}
