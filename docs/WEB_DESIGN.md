# Geocentric chat refinement

The owner's three supplied design references describe ChatGPT, Perplexity and Pika.
They are visual references, not instructions to reproduce those companies' content,
authentication screens, or service capabilities.

The interface keeps ChatGPT's narrow conversation column and quiet navigation,
uses Perplexity's warm-paper reading surface in light mode, and takes Pika's flat
charcoal contrast for dark mode. The original Geocentric wordmark and mascot remain
the identity. Thin borders, a system font, consistent spacing and limited sage
accents connect both themes. No external font or image service is required.

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

The six Node/jsdom unit tests in `tests/web_ui.test.cjs` cover starter prompts,
conversation/draft restoration, rename/delete/undo, safe Markdown, streaming node
preservation, and settings. These are headless DOM unit tests, not a browser visual
review. The `web-ui` CI job installs its test dependency separately; the shipped
Python application still needs no Node runtime.

```bash
npm install --prefix /tmp/geocentric-ui-tests --no-save --no-package-lock jsdom@26.1.0
NODE_PATH=/tmp/geocentric-ui-tests/node_modules node --test tests/web_ui.test.cjs
```
