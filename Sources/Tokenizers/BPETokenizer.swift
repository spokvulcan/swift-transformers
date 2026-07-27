//
//  BPETokenizer.swift
//  CoreMLBert
//
//  Created by Julien Chaumond on 18/07/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//

import Foundation
import Hub

/// A pair of byte/token strings used in Byte-Pair Encoding (BPE) merge operations.
struct BytePair: Hashable, Sendable {
    let a: String
    let b: String
    init(_ a: String, _ b: String) {
        self.a = a
        self.b = b
    }

    init(tuple: [String]) {
        a = tuple[0]
        b = tuple[1]
    }

    static func == (lhs: BytePair, rhs: BytePair) -> Bool {
        lhs.a == rhs.a && lhs.b == rhs.b
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(a)
        hasher.combine(b)
    }
}

/// Minimal binary min-heap. `push` and `pop` are O(log n).
/// Used by `BPETokenizer.bpe(token:)` for the priority-queue BPE merge loop,
/// as the reference implementation does, see https://github.com/huggingface/tokenizers/blob/b58227c7f1ccf8b73ee2268354336da56d91e492/tokenizers/src/models/bpe/word.rs
private struct MinHeap<Element: Comparable> {
    private var storage: [Element] = []

    var isEmpty: Bool { storage.isEmpty }

    mutating func reserveCapacity(_ n: Int) {
        storage.reserveCapacity(n)
    }

    mutating func push(_ element: Element) {
        storage.append(element)
        var i = storage.count - 1
        while i > 0 {
            let parent = (i - 1) / 2
            if storage[parent] <= storage[i] { break }
            storage.swapAt(parent, i)
            i = parent
        }
    }

    mutating func pop() -> Element? {
        guard !storage.isEmpty else { return nil }
        let top = storage[0]
        let last = storage.removeLast()
        if storage.isEmpty { return top }
        storage[0] = last
        var i = 0
        let n = storage.count
        while true {
            let l = 2 * i + 1
            let r = 2 * i + 2
            var smallest = i
            if l < n, storage[l] < storage[smallest] { smallest = l }
            if r < n, storage[r] < storage[smallest] { smallest = r }
            if smallest == i { break }
            storage.swapAt(i, smallest)
            i = smallest
        }
        return top
    }
}

/// Heap entry for the BPE merge priority queue. Lower `rank` wins; ties break
/// on leftmost `left` index, matching `huggingface/tokenizers` semantics.
private struct BPEMergeCandidate: Comparable {
    let rank: Int
    let left: Int

    static func < (lhs: Self, rhs: Self) -> Bool {
        if lhs.rank != rhs.rank { return lhs.rank < rhs.rank }
        return lhs.left < rhs.left
    }
}

/// A Byte-Pair Encoding (BPE) tokenizer implementation.
///
/// BPE tokenizers learn to merge the most frequently occurring pairs of characters
/// or character sequences. This implementation supports various BPE-based models
/// including GPT-2, RoBERTa, and other transformer models.
class BPETokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    let bpeRanks: [BytePair: Int]
    private let tokensToIds: [NSString: Int]
    private let idsToTokens: [Int: NSString]

    /// The total number of tokens in the vocabulary.
    var vocabCount: Int { tokensToIds.count }

    /// The beginning-of-sequence token string, if defined.
    let bosToken: String?

    /// The numeric ID of the beginning-of-sequence token, if defined.
    let bosTokenId: Int?

    /// The end-of-sequence token string, if defined.
    let eosToken: String?

    /// The numeric ID of the end-of-sequence token, if defined.
    let eosTokenId: Int?

    /// The unknown token string used for out-of-vocabulary words.
    let unknownToken: String?

    /// The numeric ID of the unknown token.
    let unknownTokenId: Int?

    /// Whether consecutive unknown tokens should be fused together.
    let fuseUnknownTokens: Bool

    /// Byte-keyed lookup tables, derived 1:1 from `bpeRanks` / `tokensToIds`.
    ///
    /// Built eagerly in `init` and held in a `let`, NOT lazily behind a
    /// double-checked lock. `BPETokenizer` is `@unchecked Sendable` and shared
    /// across the callers of `encode`, and an unsynchronized fast-path read of
    /// a lazily-assigned `var` has no acquire semantics: on a weakly-ordered
    /// machine (every Apple Silicon target) a reader can observe the published
    /// object pointer before the eight array buffers it points at are visible.
    /// Both source dictionaries are `let`s fully populated by the end of
    /// `init`, so there is nothing to defer — and the table build is
    /// per-tokenizer-load, which C23 measured as fully hidden behind the weight
    /// load (`async let`).
    private let byteTables: BytePairTables

    static func mergesFromConfig(_ config: Config?) -> [[String]]? {
        guard let config else { return nil }

        if let merges = config.array() {
            return merges.reduce(into: [[String]]()) { result, element in
                if let val: [String] = element.get() { // New format (pushed with tokenizers >= 0.20.0): each merge is a list of 2 items
                    result.append(val)
                }
                if let val: String = element.get() { // legacy
                    result.append(val.unicodeScalars.split(separator: " ", omittingEmptySubsequences: false).map { String($0) })
                }
            }
        }

        return nil
    }

    /// Initializes a BPE tokenizer from configuration data.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data containing vocabulary and merges
    ///   - addedTokens: Additional tokens to include in the vocabulary
    /// - Throws: `TokenizerError` if required configuration is missing
    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        guard let merges = Self.mergesFromConfig(tokenizerData.model.merges) else { fatalError("BPETokenizer requires merges") }
        guard let vocab = tokenizerData.model.vocab.dictionary() else {
            throw TokenizerError.missingVocab
        }
        var bpeRanks: [BytePair: Int] = [:]
        for (i, merge) in merges.enumerated() {
            let bp = BytePair(tuple: merge)
            bpeRanks[bp] = i
        }
        self.bpeRanks = bpeRanks

        let addedTokens = addedTokens.reduce(into: [BinaryDistinctString: Config]()) { result, element in
            result[BinaryDistinctString(element.key)] = .init(element.value)
        }
        tokensToIds = vocab.merging(addedTokens) { $1 }.reduce(into: [NSString: Int]()) { result, element in
            result[element.key.nsString] = element.value.integer()
        }

        idsToTokens = tokensToIds.reduce(into: [Int: NSString]()) { result, element in
            result[element.value] = element.key
        }

        // Populate tokens
        if let unknownToken = TokenizerModel.unknownToken(from: tokenizerConfig) {
            self.unknownToken = unknownToken
            unknownTokenId = tokensToIds[unknownToken as NSString]
        } else {
            unknownToken = nil
            unknownTokenId = nil
        }

        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken == nil ? nil : tokensToIds[eosToken! as NSString]

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken == nil ? nil : tokensToIds[bosToken! as NSString]

        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)

        // Both source dictionaries are complete: derive the byte-keyed tables
        // now so the hot paths read an immutable `let`.
        byteTables = BytePairTables(bpeRanks: bpeRanks, tokensToIds: tokensToIds)
    }

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID, or the unknown token ID if not found
    func convertTokenToId(_ token: String) -> Int? {
        // Byte-keyed lookup over the exact contents of `tokensToIds`; same
        // result as the NSString-bridged subscript, without bridging.
        byteTables.id(for: token) ?? unknownTokenId
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string, or nil if the ID is invalid
    func convertIdToToken(_ id: Int) -> String? {
        idsToTokens[id] as String?
    }

    /// Cached `<0x%02X>` byte fallback strings, indexed by byte value.
    private static let hexaEncoderTable: [String] = (0..<256).map { String(format: "<0x%02X>", $0) }

    func byteEncode(text: String) -> [String] {
        var result: [String] = []
        enumerateRegexTokens(in: text, with: byteLevelPreTokenizeRegex) { token in
            var encoded = ""
            encoded.reserveCapacity(token.utf8.count)
            for byte in token.utf8 {
                encoded.append(byteEncoderTable[Int(byte)])
            }
            result.append(encoded)
        }
        return result
    }

    func hexaEncode(text: String) -> [String] {
        var result: [String] = []
        enumerateRegexTokens(in: text, with: byteLevelPreTokenizeRegex) { token in
            for byte in token.utf8 {
                result.append(Self.hexaEncoderTable[Int(byte)])
            }
        }
        return result
    }

    /// Applies the BPE merge sequence to `token` and returns the resulting pieces.
    func bpe(token: String) -> [String] {
        bpeByteNative(token: token)
    }

    /// Byte-native implementation: the SAME algorithm with the SAME
    /// merge order — initial symbols are one per Unicode scalar (byte ranges
    /// into a single UTF-8 buffer instead of one allocated String per
    /// scalar), merges extend a byte range instead of concatenating Strings,
    /// and rank lookups run against the byte-keyed `BytePairTables` (identical
    /// contents to `bpeRanks`, including the `(rank, left)` heap tie-break
    /// and the stale-entry re-check).
    private func bpeByteNative(token: String) -> [String] {
        let tables = byteTables
        let buf = Array(token.utf8)

        // Initial symbol ranges: one per Unicode scalar — the same boundaries
        // as `token.unicodeScalars.map { String($0) }`. `buf` comes from
        // `String.utf8`, so it is always well-formed and the leading-byte width
        // never runs past the end; the `min` keeps a malformed buffer (a
        // continuation byte read as a lead, say) inside bounds rather than
        // trapping on the slice below.
        var ranges: [(start: Int, end: Int)] = []
        ranges.reserveCapacity(buf.count / 2 + 2)
        var idx = 0
        while idx < buf.count {
            let b0 = buf[idx]
            let width = b0 < 0x80 ? 1 : (b0 < 0xE0 ? 2 : (b0 < 0xF0 ? 3 : 4))
            let end = min(idx + width, buf.count)
            ranges.append((idx, end))
            idx = end
        }

        let initialCount = ranges.count
        if initialCount <= 1 {
            return initialCount == 0 ? [] : [token]
        }

        var prevIndex = Array(repeating: -1, count: initialCount)
        var nextIndex = Array(repeating: -1, count: initialCount)
        var alive = Array(repeating: true, count: initialCount)
        for i in 0..<initialCount {
            prevIndex[i] = i - 1
            nextIndex[i] = (i == initialCount - 1) ? -1 : i + 1
        }

        // Rank of the merge pair (ranges[i], ranges[j]); ranges are always
        // contiguous, so the pair key is buf[aStart..<bEnd] split at aEnd.
        @inline(__always)
        func pairRank(_ i: Int, _ j: Int) -> Int? {
            tables.rank(in: buf, aStart: ranges[i].start, aEnd: ranges[i].end, bEnd: ranges[j].end)
        }

        var heap = MinHeap<BPEMergeCandidate>()
        heap.reserveCapacity(initialCount)

        func enqueue(left: Int) {
            let right = nextIndex[left]
            guard right != -1, alive[left], alive[right] else { return }
            if let rank = pairRank(left, right) {
                heap.push(BPEMergeCandidate(rank: rank, left: left))
            }
        }

        for i in 0..<(initialCount - 1) {
            enqueue(left: i)
        }

        while let top = heap.pop() {
            let i = top.left
            guard alive[i] else { continue }
            let j = nextIndex[i]
            guard j != -1, alive[j] else { continue }
            // Validate the entry is not stale: the pair at (i, j) must still
            // have exactly the rank recorded when it was enqueued.
            guard let actualRank = pairRank(i, j), actualRank == top.rank else {
                continue
            }

            // Absorb symbol j into symbol i (byte-range extension; identical
            // to `symbols[i] = symbols[i] + symbols[j]`).
            ranges[i].end = ranges[j].end
            let k = nextIndex[j]
            nextIndex[i] = k
            if k != -1 { prevIndex[k] = i }
            alive[j] = false

            if prevIndex[i] != -1 {
                enqueue(left: prevIndex[i])
            }
            enqueue(left: i)
        }

        // Walk the surviving ranges from the head (position 0 is never
        // absorbed: merges always move bytes from `next` into `current`).
        var pieces: [String] = []
        pieces.reserveCapacity(initialCount)
        var cursor = 0
        while cursor != -1 {
            pieces.append(String(decoding: buf[ranges[cursor].start..<ranges[cursor].end], as: UTF8.self))
            cursor = nextIndex[cursor]
        }
        return pieces
    }

    /// Tokenizes input text using the BPE algorithm.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of BPE token strings
    func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        let bpeTokens = bpe(token: text)
        for token in bpeTokens {
            if convertTokenToId(token) != unknownTokenId {
                tokens.append(token)
            } else {
                // TODO: if config.byte_fallback is False, append the unknown token instead
                tokens.append(contentsOf: hexaEncode(text: token))
            }
        }
        return tokens
    }
}


/// Byte-keyed lookup tables derived 1:1 from `bpeRanks` and `tokensToIds`:
/// merge-pair UTF-8 bytes (+ split point) -> merge rank, and token UTF-8
/// bytes -> token id. Open addressing with linear probing; lookups hash raw
/// UTF-8 bytes (FNV-1a) instead of Swift String keys, so no String hashing
/// (NFC walk), no NSString bridging, and no per-lookup allocation. Keys are
/// verified with a full byte comparison.
///
/// **The pair table's match semantics are byte-exact, which is a DELIBERATE
/// narrowing of what it replaces.** `tokensToIds` is `[NSString: Int]`, and
/// NSString compares UTF-16 code units, so `id(for:)` is exactly the old
/// subscript. `bpeRanks` is `[BytePair: Int]` and `BytePair` holds Swift
/// `String`s (see its declaration), so the dictionary probe it replaces
/// compared under Unicode CANONICAL EQUIVALENCE: an NFD merge key would match
/// NFC text and vice versa, and two normalization variants of one merge
/// collapsed onto a single dictionary slot (last write wins). `rank(in:...)`
/// matches bytes.
///
/// Byte-exact is the intended semantics — it is what `huggingface/tokenizers`
/// and `tiktoken` do, the merge table is a byte-level artifact, and
/// normalization is the normalizer's job, not the BPE inner loop's. It is
/// unobservable for byte-level BPE vocabs (the GPT-2 byte->unicode map emits no
/// combining marks, so no vocab key has a distinct normalization form), which
/// is why the C24 gate saw 88/88 byte-identical items over 6.7M tokens on both
/// PARO tokenizers. It CAN change output on a non-byte-level BPE vocab that
/// carries mixed normalization forms — there, this is a fix, not a
/// regression.
private final class BytePairTables {
    private struct PairEntry {
        var hash: UInt64
        var blobOffset: Int32
        var lenA: Int32
        var lenB: Int32
        var rank: Int32
    }

    private struct IdEntry {
        var hash: UInt64
        var blobOffset: Int32
        var len: Int32
        var id: Int32
    }

    private let pairBlob: [UInt8]
    private let pairEntries: [PairEntry]
    private let pairSlots: [Int32]  // entry index + 1; 0 = empty
    private let pairMask: Int

    private let idBlob: [UInt8]
    private let idEntries: [IdEntry]
    private let idSlots: [Int32]
    private let idMask: Int

    @inline(__always) private static func fnvContinue(_ h: UInt64, _ b: UInt8) -> UInt64 {
        (h ^ UInt64(b)) &* 0x0000_0100_0000_01B3
    }

    init(bpeRanks: [BytePair: Int], tokensToIds: [NSString: Int]) {
        var blob: [UInt8] = []
        var entries: [PairEntry] = []
        entries.reserveCapacity(bpeRanks.count)
        blob.reserveCapacity(bpeRanks.count * 8)
        for (pair, rank) in bpeRanks {
            let offset = blob.count
            var h: UInt64 = 0xCBF2_9CE4_8422_2325
            for b in pair.a.utf8 { h = BytePairTables.fnvContinue(h, b) }
            for b in pair.b.utf8 { h = BytePairTables.fnvContinue(h, b) }
            h = BytePairTables.fnvContinue(h, UInt8(truncatingIfNeeded: pair.a.utf8.count))
            h = BytePairTables.fnvContinue(h, UInt8(truncatingIfNeeded: pair.a.utf8.count >> 8))
            blob.append(contentsOf: pair.a.utf8)
            blob.append(contentsOf: pair.b.utf8)
            entries.append(PairEntry(
                hash: h,
                blobOffset: Int32(offset),
                lenA: Int32(pair.a.utf8.count),
                lenB: Int32(pair.b.utf8.count),
                rank: Int32(rank)))
        }
        pairBlob = blob
        pairEntries = entries
        (pairSlots, pairMask) = BytePairTables.buildSlots(hashes: entries.map { $0.hash })

        var iblob: [UInt8] = []
        var ientries: [IdEntry] = []
        ientries.reserveCapacity(tokensToIds.count)
        iblob.reserveCapacity(tokensToIds.count * 6)
        for (key, id) in tokensToIds {
            let str = key as String
            let offset = iblob.count
            var h: UInt64 = 0xCBF2_9CE4_8422_2325
            for b in str.utf8 { h = BytePairTables.fnvContinue(h, b) }
            h = BytePairTables.fnvContinue(h, 0)
            h = BytePairTables.fnvContinue(h, 0)
            iblob.append(contentsOf: str.utf8)
            ientries.append(IdEntry(
                hash: h,
                blobOffset: Int32(offset),
                len: Int32(str.utf8.count),
                id: Int32(id)))
        }
        idBlob = iblob
        idEntries = ientries
        (idSlots, idMask) = BytePairTables.buildSlots(hashes: ientries.map { $0.hash })
    }

    private static func buildSlots(hashes: [UInt64]) -> ([Int32], Int) {
        var size = 16
        while size < hashes.count * 2 { size *= 2 }
        var slots = [Int32](repeating: 0, count: size)
        let mask = size - 1
        for (i, h) in hashes.enumerated() {
            var slot = Int(h & UInt64(mask))
            while slots[slot] != 0 { slot = (slot + 1) & mask }
            slots[slot] = Int32(i + 1)
        }
        return (slots, mask)
    }

    @inline(__always)
    private static func bytesEqual(_ a: [UInt8], _ aOff: Int, _ b: [UInt8], _ bOff: Int, _ n: Int) -> Bool {
        for i in 0 ..< n where a[aOff + i] != b[bOff + i] { return false }
        return true
    }

    /// Rank of the merge pair whose left part is buf[aStart..<aEnd] and right
    /// part is buf[aEnd..<bEnd]; nil if the pair has no merge.
    func rank(in buf: [UInt8], aStart: Int, aEnd: Int, bEnd: Int) -> Int? {
        let lenA = aEnd - aStart
        let lenB = bEnd - aEnd
        var h: UInt64 = 0xCBF2_9CE4_8422_2325
        var i = aStart
        while i < aEnd { h = BytePairTables.fnvContinue(h, buf[i]); i += 1 }
        while i < bEnd { h = BytePairTables.fnvContinue(h, buf[i]); i += 1 }
        h = BytePairTables.fnvContinue(h, UInt8(truncatingIfNeeded: lenA))
        h = BytePairTables.fnvContinue(h, UInt8(truncatingIfNeeded: lenA >> 8))
        var slot = Int(h & UInt64(pairMask))
        while true {
            let s = pairSlots[slot]
            if s == 0 { return nil }
            let e = pairEntries[Int(s) - 1]
            if e.hash == h, Int(e.lenA) == lenA, Int(e.lenB) == lenB,
                BytePairTables.bytesEqual(pairBlob, Int(e.blobOffset), buf, aStart, lenA + lenB)
            {
                return Int(e.rank)
            }
            slot = (slot + 1) & pairMask
        }
    }

    /// Id of the token with exactly these UTF-8 bytes; nil if not in vocab.
    func id(for token: String) -> Int? {
        let utf8 = token.utf8
        var h: UInt64 = 0xCBF2_9CE4_8422_2325
        for b in utf8 { h = BytePairTables.fnvContinue(h, b) }
        h = BytePairTables.fnvContinue(h, 0)
        h = BytePairTables.fnvContinue(h, 0)
        var slot = Int(h & UInt64(idMask))
        while true {
            let s = idSlots[slot]
            if s == 0 { return nil }
            let e = idEntries[Int(s) - 1]
            if e.hash == h, Int(e.len) == utf8.count {
                var matched = true
                var i = Int(e.blobOffset)
                for b in utf8 {
                    if idBlob[i] != b { matched = false; break }
                    i += 1
                }
                if matched { return Int(e.id) }
            }
            slot = (slot + 1) & idMask
        }
    }
}

