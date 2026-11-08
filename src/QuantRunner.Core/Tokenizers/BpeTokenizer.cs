using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace QuantRunner.Core.Tokenizers;

/// <summary>
/// HuggingFace byte-level BPE tokenizer (GPT-2/Llama style), loaded from tokenizer.json.
/// Implements the standard pipeline: byte-to-unicode remap -> GPT-2 pre-tokenization regex
/// -> rank-based BPE merges -> vocab ids, plus the inverse for detokenization.
/// Zero dependencies; typical encode speed is >1M chars/s on modern hardware.
/// </summary>
public sealed partial class BpeTokenizer
{
    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _inv = new();
    private readonly Dictionary<string, int> _ranks = new();
    private readonly Dictionary<string, int[]> _cache = new(StringComparer.Ordinal);
    private readonly Dictionary<int, (string Token, bool Special)> _added = new();
    private readonly Dictionary<byte, char> _byteToChar = new();
    private readonly Dictionary<char, byte> _charToByte = new();

    [GeneratedRegex(
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")]
    private static partial Regex Gpt2Split();

    public int VocabSize => _vocab.Count;
    public int EosId { get; private set; } = 0;
    public int BosId { get; private set; } = 1;

    public static BpeTokenizer Load(string tokenizerJsonPath) => LoadJson(File.ReadAllBytes(tokenizerJsonPath));

    /// <summary>Parse a tokenizer.json from memory — used when the tokenizer is embedded in a
    /// model file (GGUF metadata, or the gguf_kvs section a .qw imported from GGUF carries).</summary>
    public static BpeTokenizer LoadJson(byte[] tokenizerJsonBytes)
    {
        var tok = new BpeTokenizer();
        using var doc = JsonDocument.Parse(tokenizerJsonBytes);
        var root = doc.RootElement;

        // ---- vocab + merges ----
        var model = root.GetProperty("model");
        foreach (var p in model.GetProperty("vocab").EnumerateObject())
        {
            int id = p.Value.GetInt32();
            tok._vocab[p.Name] = id;
            tok._inv[id] = p.Name;
        }

        int nextRank = 0;
        if (model.TryGetProperty("merges", out var mergesEl) && mergesEl.ValueKind == JsonValueKind.Array)
        {
            foreach (var m in mergesEl.EnumerateArray())
            {
                if (m.ValueKind == JsonValueKind.String)
                {
                    string s = m.GetString()!;
                    int sp = s.IndexOf(' ');
                    if (sp > 0) tok._ranks[s] = nextRank++;
                }
                else if (m.ValueKind == JsonValueKind.Array && m.GetArrayLength() == 2)
                {
                    tok._ranks[m[0].GetString()! + " " + m[1].GetString()!] = nextRank++;
                }
            }
        }

        // ---- special / added tokens ----
        if (root.TryGetProperty("added_tokens", out var addedEl) && addedEl.ValueKind == JsonValueKind.Array)
        {
            foreach (var at in addedEl.EnumerateArray())
            {
                int id = at.GetProperty("id").GetInt32();
                string content = at.GetProperty("content").GetString()!;
                bool special = !at.TryGetProperty("special", out var spEl) || spEl.ValueKind != JsonValueKind.True || spEl.GetBoolean();
                tok._added[id] = (content, special);
                tok._vocab.TryAdd(content, id);
                tok._inv.TryAdd(id, content);
            }
        }

        // ---- conventional ids ----
        if (root.TryGetProperty("eos_token", out var eosEl))
            tok.EosId = tok._vocab.TryGetValue(eosEl.GetString() ?? "", out int e) ? e : 0;
        if (root.TryGetProperty("bos_token", out var bosEl))
            tok.BosId = tok._vocab.TryGetValue(bosEl.GetString() ?? "", out int b) ? b : 1;

        tok.BuildByteMap();
        return tok;
    }

    /// <summary>GPT-2 bytes-to-unicode: printable latin chars map to themselves, the rest to 0x100+.</summary>
    private void BuildByteMap()
    {
        List<int> bs = new();
        for (int b = '!' ; b <= '~'; b++) bs.Add(b);
        for (int b = 0xA1; b <= 0xAC; b++) bs.Add(b);
        for (int b = 0xAE; b <= 0xFF; b++) bs.Add(b);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b)) bs.Add(b);
        }
        foreach (int b in bs)
        {
            char c = (char)b;
            if (c < '!' || c > '~') c = (char)(256 + n++);
            _byteToChar[(byte)b] = c;
            _charToByte[c] = (byte)b;
        }
    }

    // ------------------------------------------------------------------ encode

    public int[] Encode(string text)
    {
        var ids = new List<int>(text.Length / 4 + 8);

        // added/special tokens win over BPE (longest match first, in id order)
        int i = 0;
        while (i < text.Length)
        {
            int matchId = -1, matchLen = 0;
            foreach (var (id, (content, special)) in _added)
            {
                if (!special && content.Length <= 1) continue;
                if (content.Length > matchLen && string.CompareOrdinal(text, i, content, 0, content.Length) == 0)
                { matchId = id; matchLen = content.Length; }
            }
            if (matchId >= 0)
            {
                ids.Add(matchId);
                i += matchLen;
                continue;
            }
            int j = i;
            foreach (var piece in Gpt2Split().EnumerateMatches(text, i))
            {
                j = piece.Index + piece.Length;
                break;
            }
            if (j <= i) j = i + 1;
            EncodePiece(text[i..j], ids);
            i = j;
        }
        return ids.ToArray();
    }

    private void EncodePiece(string piece, List<int> ids)
    {
        // UTF-8 bytes -> remapped chars -> BPE merges
        var bytes = Encoding.UTF8.GetBytes(piece);
        var sb = new StringBuilder(bytes.Length);
        foreach (var b in bytes) sb.Append(_byteToChar[b]);
        string word = sb.ToString();

        if (_cache.TryGetValue(word, out var cached)) { ids.AddRange(cached); return; }

        // start from individual characters, then apply rank-ordered merges
        var parts = new List<string>(word.Length);
        foreach (char c in word) parts.Add(c.ToString());
        while (parts.Count > 1)
        {
            int bestRank = int.MaxValue, bestIdx = -1;
            for (int p = 0; p + 1 < parts.Count; p++)
            {
                if (_ranks.TryGetValue(parts[p] + " " + parts[p + 1], out int r) && r < bestRank)
                { bestRank = r; bestIdx = p; }
            }
            if (bestIdx < 0) break;
            parts[bestIdx] += parts[bestIdx + 1];
            parts.RemoveAt(bestIdx + 1);
        }

        var result = new int[parts.Count];
        for (int p = 0; p < parts.Count; p++)
        {
            if (_vocab.TryGetValue(parts[p], out int id)) result[p] = id;
            else result[p] = UnknownId;
        }
        _cache.TryAdd(word, result);
        ids.AddRange(result);
    }

    public int UnknownId { get; private set; } = 0;

    // ------------------------------------------------------------------ decode

    public string Decode(int[] ids) => Decode((ReadOnlySpan<int>)ids);

    public string Decode(ReadOnlySpan<int> ids)
    {
        var sb = new List<byte>(ids.Length * 4);
        foreach (int id in ids)
        {
            if (_added.TryGetValue(id, out var added) && added.Special) continue; // don't emit specials
            if (!_inv.TryGetValue(id, out string tok)) continue;
            foreach (char c in tok)
            {
                if (_charToByte.TryGetValue(c, out byte b)) sb.Add(b);
                // else: literal char (special token text) — skip in byte stream
            }
        }
        return Encoding.UTF8.GetString(sb.ToArray());
    }

    public string IdToToken(int id) => _inv.TryGetValue(id, out var t) ? t : "";

    /// <summary>Human-readable text of a single token (for incremental streaming output).</summary>
    public string DecodeToken(int id)
    {
        if (!_inv.TryGetValue(id, out string tok)) return "";
        if (_added.TryGetValue(id, out var a) && a.Special) return tok;
        var bytes = new List<byte>(tok.Length);
        foreach (char c in tok)
            if (_charToByte.TryGetValue(c, out byte b)) bytes.Add(b);
        return Encoding.UTF8.GetString(bytes.ToArray());
    }
}
