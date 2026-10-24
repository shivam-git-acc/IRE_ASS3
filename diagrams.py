"""Inline SVG diagram for the report.

One diagram, because one mechanism is genuinely hard to assemble from prose: where the
tune/eval query split sits, and why recall and throughput are measured by two different
passes over the same index.

Colours match the plots — blue HNSW, green IVF-Flat, purple IVF-PQ, crimson exact.
"""

BLUE, GREEN, PURPLE, CRIMSON, INK, MUTE = (
    "#2563eb", "#059669", "#9333ea", "#dc143c", "#1c1c1c", "#6b6b6b")


def protocol_diagram():
    """Where the tune/eval split sits, and why recall and timing take different paths."""
    s = ['<svg viewBox="0 0 370 196" role="img" aria-label="Measurement protocol: the search '
         'parameter is chosen on the tune half of the queries, and the reported recall and '
         'throughput are measured on the eval half by two separate passes." '
         'style="width:100%;height:auto">',
         '<defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5"'
         ' markerHeight="5" orient="auto-start-reverse">'
         '<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>']
    s.append(f'<rect x="92" y="4" width="186" height="20" rx="2" fill="none" stroke="{INK}" '
             f'stroke-width="1"/>')
    s.append(f'<text x="185" y="18" font-size="10" text-anchor="middle" fill="{INK}">'
             '10,000 held-out queries</text>')
    s.append(f'<rect x="92" y="38" width="88" height="20" rx="2" fill="{MUTE}" '
             f'fill-opacity="0.12" stroke="{MUTE}" stroke-width="0.9"/>')
    s.append(f'<text x="136" y="52" font-size="9.5" text-anchor="middle" fill="{INK}">'
             'tune 0–4999</text>')
    s.append(f'<rect x="190" y="38" width="88" height="20" rx="2" fill="{BLUE}" '
             f'fill-opacity="0.12" stroke="{BLUE}" stroke-width="0.9"/>')
    s.append(f'<text x="234" y="52" font-size="9.5" text-anchor="middle" fill="{INK}">'
             'eval 5000–9999</text>')
    for x in (136, 234):
        s.append(f'<g color="{INK}"><line x1="{x}" y1="26" x2="{x}" y2="35" '
                 f'stroke="currentColor" stroke-width="1" marker-end="url(#ah2)"/></g>')
    s.append(f'<rect x="12" y="76" width="140" height="30" rx="2" fill="none" '
             f'stroke="{MUTE}" stroke-width="0.9"/>')
    s.append(f'<text x="82" y="89" font-size="9.5" text-anchor="middle" fill="{INK}">'
             'parameter screen</text>')
    s.append(f'<text x="82" y="100" font-size="8.5" text-anchor="middle" fill="{MUTE}">'
             'recall only · stop at 0.95</text>')
    s.append(f'<g color="{MUTE}"><path d="M120,60 L120,68 L82,68 L82,73" fill="none" '
             f'stroke="currentColor" stroke-width="1" marker-end="url(#ah2)"/></g>')
    s.append(f'<rect x="140" y="124" width="92" height="22" rx="2" fill="none" '
             f'stroke="{INK}" stroke-width="1.1"/>')
    s.append(f'<text x="186" y="139" font-size="10" text-anchor="middle" fill="{INK}">'
             'the one index</text>')
    s.append(f'<g color="{MUTE}"><path d="M82,109 L82,135 L135,135" fill="none" '
             f'stroke="currentColor" stroke-width="1" marker-end="url(#ah2)"/></g>')
    s.append(f'<text x="86" y="121" font-size="8.5" fill="{MUTE}">chosen efSearch / nprobe'
             '</text>')
    s.append(f'<g color="{BLUE}"><path d="M234,60 L234,116 L210,116 L210,121" fill="none" '
             f'stroke="currentColor" stroke-width="1" marker-end="url(#ah2)"/></g>')
    s.append(f'<rect x="12" y="160" width="160" height="30" rx="2" fill="{GREEN}" '
             f'fill-opacity="0.10" stroke="{GREEN}" stroke-width="0.9"/>')
    s.append(f'<text x="92" y="173" font-size="9.5" text-anchor="middle" fill="{INK}">'
             'recall@10</text>')
    s.append(f'<text x="92" y="184" font-size="8.5" text-anchor="middle" fill="{MUTE}">'
             'batched · 4 threads</text>')
    s.append(f'<rect x="186" y="160" width="172" height="30" rx="2" fill="{BLUE}" '
             f'fill-opacity="0.10" stroke="{BLUE}" stroke-width="0.9"/>')
    s.append(f'<text x="272" y="173" font-size="9.5" text-anchor="middle" fill="{INK}">'
             'throughput</text>')
    s.append(f'<text x="272" y="184" font-size="8.5" text-anchor="middle" fill="{MUTE}">'
             '1 query at a time · 1 thread · ×3</text>')
    s.append(f'<g color="{GREEN}"><path d="M168,148 L120,148 L120,156" fill="none" '
             f'stroke="currentColor" stroke-width="1" marker-end="url(#ah2)"/></g>')
    s.append(f'<g color="{BLUE}"><path d="M204,148 L252,148 L252,156" fill="none" '
             f'stroke="currentColor" stroke-width="1" marker-end="url(#ah2)"/></g>')
    s.append("</svg>")
    return "\n".join(s)
