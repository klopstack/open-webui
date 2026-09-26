// @vitest-environment happy-dom
import { describe, expect, it } from 'vitest';

import { htmlToMarkdown, markdownToHtml } from './markdown';

/**
 * OCR-like fixture: the shapes chandra-ocr markdown actually contains
 * (headings, emphasis, lists, task lists, GFM tables, fenced code,
 * blockquote, hr, links, inline code).
 */
const FIXTURE_MD = `# Title

Some intro with **bold**, *italic*, ~~strike~~ and \`inline code\`.

## Section

- alpha
- beta
  - nested

1. first
2. second

- [x] done item
- [ ] open item

| Col A | Col B |
| ----- | ----- |
| a1 | b1 |
| a2 | b2 |

\`\`\`python
print("hello")
\`\`\`

> quoted line

---

See [the docs](https://example.com/page) for details.
`;

describe('markdownToHtml (import)', () => {
	it('renders GFM tables as real <table> elements', () => {
		const html = markdownToHtml(FIXTURE_MD);
		expect(html).toContain('<table>');
		// header row + 2 body rows
		const rows = html.match(/<tr>/g) ?? [];
		expect(rows.length).toBe(3);
		expect(html).toContain('<th');
		expect(html).toContain('>Col A<');
		expect(html).toContain('>b2<');
	});

	it('marks task lists with data-type and data-checked attrs', () => {
		const html = markdownToHtml(FIXTURE_MD);
		expect(html).toContain('data-type="taskList"');
		expect(html).toContain('data-checked="true"');
		expect(html).toContain('data-checked="false"');
	});

	it('preserves heading levels, emphasis, links, code, blockquote, hr', () => {
		const html = markdownToHtml(FIXTURE_MD);
		expect(html).toContain('<h1>Title</h1>');
		expect(html).toContain('<h2>Section</h2>');
		expect(html).toContain('<strong>bold</strong>');
		expect(html).toContain('<em>italic</em>');
		expect(html).toContain('<del>strike</del>');
		expect(html).toContain('<code>inline code</code>');
		expect(html).toContain('<blockquote>');
		expect(html).toContain('<hr');
		expect(html).toContain('href="https://example.com/page"');
		expect(html).toContain('language-python');
	});
});

describe('htmlToMarkdown (export)', () => {
	it('exports tables as GFM pipe tables', () => {
		const md = htmlToMarkdown(markdownToHtml(FIXTURE_MD));
		expect(md).toContain('| Col A | Col B |');
		expect(md).toMatch(/\| --- \| --- \|/);
		expect(md).toContain('| a1 | b1 |');
		expect(md).toContain('| a2 | b2 |');
	});

	it('exports task lists with checkbox markers', () => {
		const md = htmlToMarkdown(markdownToHtml(FIXTURE_MD));
		expect(md).toContain('- [x] done item');
		expect(md).toContain('- [ ] open item');
	});

	it('exports headings, emphasis, code fence, blockquote, hr, link', () => {
		const md = htmlToMarkdown(markdownToHtml(FIXTURE_MD));
		expect(md).toContain('# Title');
		expect(md).toContain('## Section');
		expect(md).toContain('**bold**');
		// turndown exports em with underscores (valid markdown)
		expect(md).toContain('_italic_');
		expect(md).toContain('~~strike~~');
		expect(md).toContain('```python');
		expect(md).toContain('print("hello")');
		expect(md).toContain('> quoted line');
		expect(md).toMatch(/^-{3,}$/m);
		expect(md).toContain('[the docs](https://example.com/page)');
	});
});

describe('round trip (export ∘ import)', () => {
	const roundTrip = (md: string) => htmlToMarkdown(markdownToHtml(md));

	it('normalizes the fixture to a stable form', () => {
		const once = roundTrip(FIXTURE_MD);
		// structural spot-checks on the normalized output
		expect(once).toContain('# Title');
		expect(once).toContain('| Col A | Col B |');
		expect(once).toContain('- [x] done item');
	});

	it('is idempotent: f(f(x)) === f(x)', () => {
		const once = roundTrip(FIXTURE_MD);
		const twice = roundTrip(once);
		expect(twice).toBe(once);
	});

	it('is idempotent on a minimal doc', () => {
		const md = '# Hi\n\nJust a paragraph.\n';
		const once = roundTrip(md);
		expect(roundTrip(once)).toBe(once);
	});
});
