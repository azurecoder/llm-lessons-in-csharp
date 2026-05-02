namespace ukazure.llm.NanoGpt;

internal sealed record DocumentAnswer(string Answer, IReadOnlyList<(string Text, int Score)> Evidence);
