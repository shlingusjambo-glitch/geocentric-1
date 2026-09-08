# Geocentric chat refinement

The owner's three supplied design references describe ChatGPT, Perplexity and Pika.
They are visual references, not instructions to reproduce those companies' content,
authentication screens, or service capabilities.

The interface keeps ChatGPT's narrow conversation column and quiet navigation,
uses neutral white reading surfaces in light mode and flat charcoal in dark mode.
The original Geocentric wordmark and mascot remain the identity. Green is confined
to artwork and restrained interaction accents, rather than tinting every surface.
Native system fonts keep controls familiar on Ubuntu, Windows and macOS.
Assistant responses and the welcome heading use local Georgia/Times serif fallbacks
for a quieter reading surface. No external font or image service is required.

The September 7 refinement follows the owner's supplied design guide: a quiet
writing workspace framed by Geocentric artwork. The composer is the primary
control, starter prompts are compact buttons rather than a repeated card grid,
and the decorative welcome eyebrow is removed. Light and dark themes share
neutral gray tokens. The send control and mobile icon controls have 44px targets;
mobile history actions remain visible without requiring hover.

The welcome screen uses restrained artwork and context-aware starter prompts.
Prompts fill an editable draft rather than immediately requesting inference.
Base models get passage beginnings; chat models get instruction-style prompts.
Conversation history supports content search, date grouping, rename, undo deletion,
and restoration of the selected conversation and its draft after reload.

During generation, only the current answer body is rendered for each chunk. Earlier
messages and sidebar controls are no longer recreated repeatedly. This reduces UI
work; it does not change model throughput, weights, training or inference numerics.
Markdown lists, headings, quotes, fenced code and HTTP(S) links use DOM text nodes;
HTML from model output is never interpreted. The mobile layout follows document
flow rather than absolute positioning of the welcome screen and composer. Keyboard
shortcuts follow the operating system, system appearance responds to theme changes,
and motion respects reduced-motion preferences.

The README uses the supplied logo at the top and a generated thumbs-up mascot next
to existing benchmark results. See [artwork provenance and prompt](assets/README.md).
This pass makes no new GPU speed claim and starts no training runs.

## Validation

September 7: 11 Node/jsdom tests pass. Actual browser review covered dark desktop
and light mobile, including 320px and 390px screenshots, theme switching and an
editable starter prompt. Overflow checks passed at 320, 375, 390, 430, 768, 1024,
1280, 1440 and 1920px. This is targeted visual review, not a complete accessibility
audit or a measured Core Web Vitals claim.

The Node/jsdom unit tests in `tests/web_ui.test.cjs` cover starter prompts,
conversation/draft restoration, rename/delete/undo, safe Markdown, streaming node
preservation, and settings. These are headless DOM unit tests, not a browser visual
review. The `web-ui` CI job installs its test dependency separately; the shipped
Python application still needs no Node runtime.

```bash
npm install --prefix /tmp/geocentric-ui-tests --no-save --no-package-lock jsdom@26.1.0
NODE_PATH=/tmp/geocentric-ui-tests/node_modules node --test tests/web_ui.test.cjs
```

## Claude / Haiku interaction refinement

Inspected the owner's Claude desktop app with Haiku 4.5, asked for a short
explanation plus a Python example, then requested an interactive counter and
verified its plus button changed the count. Observed response typography, code
presentation, the model menu, and composer placement. No Claude artwork or
generated implementation was copied into Geocentric.

The conversation title now appears in the header. The smaller composer displays
the loaded model beside Send, and the welcome mascot sits alongside its heading.
Settings has one entry in the sidebar profile; duplicate header/composer controls
were removed after owner feedback. The loaded model is an informational label,
not a pretend model switcher. Arrow keys move between messages when the conversation
or a message is focused, while leaving text editing alone.

Validation for this refinement: 13 Node/jsdom tests passed, including restored
header titles, keyboard navigation, and the single settings entry. Live local
browser review exercised inference, the narrow-layout sidebar, and opening Settings.
This changes presentation and navigation, not model quality or GPU throughput.
