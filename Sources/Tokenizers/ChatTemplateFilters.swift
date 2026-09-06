//
//  ChatTemplateFilters.swift
//
//  Jinja filters the chat-template renderer provides on top of swift-jinja's
//  built-ins, to render the way Hugging Face `transformers` does.
//

import Foundation
import Jinja

/// `tojson` with the semantics of `transformers`' chat-template rendering.
///
/// `transformers` overrides Jinja's `tojson` with
/// `json.dumps(x, ensure_ascii=False, indent=indent, separators=separators,
/// sort_keys=sort_keys)` (`utils/chat_template_utils.py`), and chat templates
/// and the models trained on their output assume that form. swift-jinja's
/// built-in differs in two ways that change the rendered text: it encodes
/// through `JSONEncoder` without `.withoutEscapingSlashes`, so `</style>`
/// renders as `<\/style>` (Python's `json.dumps` never escapes `/`), and its
/// `ensure_ascii` defaults to `true`, so `—` renders as `\u2014`. A tool call
/// re-rendered from its parsed arguments then no longer matches the text the
/// model emitted, which breaks any prompt cache keyed on the render and shows
/// the model an escaped copy of its own previous call.
///
/// This filter keeps `JSONEncoder` (and its sorted keys, which give a stable
/// render for arguments that arrive as unordered Swift dictionaries) and fixes
/// the two escaping differences: slashes are never escaped, and non-ASCII is
/// escaped only when the template asks for `ensure_ascii=True`.
///
/// Registered on the render context under the filter's name; swift-jinja
/// resolves environment-provided filters before its built-ins.
@Sendable
func transformersToJSON(
    _ args: [Jinja.Value], _ kwargs: [String: Jinja.Value], _ env: Jinja.Environment
) throws -> Jinja.Value {
    guard let value = args.first else { return .string("null") }
    let positional = Array(args.dropFirst())
    let indent = kwargs["indent"] ?? positional.first ?? .null
    let ensureASCII = kwargs["ensure_ascii"] ?? (positional.count > 1 ? positional[1] : .boolean(false))

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    if case let .int(count) = indent, count > 0 {
        encoder.outputFormatting.insert(.prettyPrinted)
    }
    guard let data = try? encoder.encode(value),
        let json = String(data: data, encoding: .utf8)
    else {
        return .string("null")
    }
    return .string(ensureASCII.isTruthy ? escapeNonASCII(json) : json)
}

/// `\uXXXX` for every UTF-16 code unit above ASCII (non-BMP scalars become
/// surrogate pairs), matching `json.dumps(ensure_ascii=True)`.
private func escapeNonASCII(_ string: String) -> String {
    var result = ""
    result.reserveCapacity(string.utf16.count)
    for codeUnit in string.utf16 {
        if codeUnit > 127 {
            result += String(format: "\\u%04x", codeUnit)
        } else if let scalar = UnicodeScalar(codeUnit) {
            result.append(Character(scalar))
        }
    }
    return result
}
