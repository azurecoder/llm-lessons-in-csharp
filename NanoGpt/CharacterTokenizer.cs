namespace ukazure.llm.NanoGpt;

internal sealed class CharacterTokenizer
{
    private readonly Dictionary<char, int> _charToId;
    private readonly Dictionary<int, char> _idToChar;

    private CharacterTokenizer(Dictionary<char, int> charToId, Dictionary<int, char> idToChar)
    {
        _charToId = charToId;
        _idToChar = idToChar;
    }

    public int VocabularySize => _charToId.Count;

    public IReadOnlyList<char> Vocabulary => _charToId.Keys.OrderBy(character => character).ToList();

    public static CharacterTokenizer FromText(string text)
    {
        var vocabulary = text
            .Distinct()
            .OrderBy(character => character)
            .ToList();

        var charToId = vocabulary
            .Select((character, index) => new { character, index })
            .ToDictionary(entry => entry.character, entry => entry.index);

        var idToChar = charToId.ToDictionary(entry => entry.Value, entry => entry.Key);

        return new CharacterTokenizer(charToId, idToChar);
    }

    public int[] Encode(string text)
    {
        return text.Select(character => _charToId[character]).ToArray();
    }

    public string Decode(IEnumerable<int> tokenIds)
    {
        return new string(tokenIds.Select(tokenId => _idToChar[tokenId]).ToArray());
    }
}
