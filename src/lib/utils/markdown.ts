/**
 * Two-way markdown <-> HTML serialization for the KB WYSIWYG editor.
 *
 * IMPORT (markdownToHtml): marked (GFM) -> HTML. The HTML is fed to TipTap's
 * ProseMirror DOMParser (never innerHTML), so it is XSS-safe. The custom
 * list/listitem renderer emits data-type="taskList" / data-checked attrs so
 * TipTap's ListKit taskList node parses (same pattern as RichTextInput).
 *
 * EXPORT (htmlToMarkdown): turndown (joplin GFM plugin) with the table /
 * task-list / single-newline rules ported from RichTextInput's proven
 * config. Round-trip is lossy-once: the first save normalizes the markdown,
 * subsequent passes are idempotent (asserted in markdown.spec.ts).
 *
 * Kept as pure functions (fresh marked/turndown instances per call) so they
 * are unit-testable without a live editor and cannot leak state into the
 * chat Markdown renderer's global marked instance.
 */
import { Marked } from 'marked';
import TurndownService from 'turndown';
import { gfm } from '@joplin/turndown-plugin-gfm';

const markedInstance = new Marked({
	breaks: true,
	gfm: true,
	renderer: {
		// marked v9 renderer signature: positional args.
		list(body: string, ordered: boolean, start: number | '') {
			const isTaskList = body.includes('data-checked=');
			if (isTaskList) {
				return `<ul data-type="taskList">${body}</ul>`;
			}
			const type = ordered ? 'ol' : 'ul';
			const startAttr = ordered && start !== 1 ? ` start="${start}"` : '';
			return `<${type}${startAttr}>${body}</${type}>`;
		},
		listitem(text: string, task: boolean, checked: boolean) {
			if (task) {
				const checkedAttr = checked ? 'true' : 'false';
				return `<li data-type="taskItem" data-checked="${checkedAttr}">${text}</li>`;
			}
			return `<li>${text}</li>`;
		}
	}
});

/** Render markdown to HTML suitable for TipTap's DOMParser. */
export const markdownToHtml = (md: string): string => {
	const html = markedInstance.parse(md ?? '') as string;
	return typeof html === 'string' ? html : '';
};

/**
 * Render TipTap HTML back to markdown.
 *
 * Verified against turndown 7.2.2 internals (Rules.add UNSHIFTs, so
 * addRule'd rules take precedence over defaults and joplin's plugin rules,
 * and findRule fires the FIRST match per node):
 *  - singleNewlineParagraphs: beats the default paragraph rule; emits \n
 *    around paragraphs instead of \n\n so code-block interiors are
 *    untouched (no destructive replaceAll needed).
 *  - tables: beats joplin's table rule and owns the whole subtree (cells
 *    are re-turndowned individually), so joplin's tableRow/tableCell never
 *    fire. tableHeaders strips th wrappers so header cells export as plain
 *    text.
 *  - taskListItems (LI[data-checked]): beats both the default listItem and
 *    joplin's INPUT-based taskListItems rule, and owns the checkbox marker;
 *    taskItemCheckbox drops the inner <input>.
 */
export const htmlToMarkdown = (html: string): string => {
	const service = new TurndownService({
		codeBlockStyle: 'fenced',
		headingStyle: 'atx'
	});
	service.escape = (string) => string;

	service.addRule('singleNewlineParagraphs', {
		filter: 'p',
		replacement: function (content) {
			return '\n' + content + '\n';
		}
	});

	service.use(gfm);

	service.addRule('tableHeaders', {
		filter: 'th',
		replacement: function (content) {
			return content;
		}
	});

	service.addRule('tables', {
		filter: 'table',
		replacement: function (content, node) {
			const rows = Array.from((node as HTMLElement).querySelectorAll('tr'));
			if (rows.length === 0) return content;

			let markdown = '\n';

			rows.forEach((row, rowIndex) => {
				const cells = Array.from(row.querySelectorAll('th, td'));
				const cellContents = cells.map((cell) => {
					let cellContent = service.turndown((cell as HTMLElement).innerHTML).trim();
					cellContent = cellContent.replace(/^\n+|\n+$/g, '');
					return cellContent;
				});

				markdown += '| ' + cellContents.join(' | ') + ' |\n';
				if (rowIndex === 0) {
					const separator = cells.map(() => '---').join(' | ');
					markdown += '| ' + separator + ' |\n';
				}
			});

			return markdown + '\n';
		}
	});

	// Deterministic hr form (turndown's default is "* * *").
	service.addRule('hr', {
		filter: 'hr',
		replacement: () => '\n\n---\n\n'
	});

	service.addRule('taskItemCheckbox', {
		filter: (node) => node.nodeName === 'INPUT' && node.getAttribute('type') === 'checkbox',
		replacement: () => ''
	});

	service.addRule('taskListItems', {
		filter: (node) =>
			node.nodeName === 'LI' &&
			(node.getAttribute('data-checked') === 'true' || node.getAttribute('data-checked') === 'false'),
		replacement: function (content, node) {
			const checked = (node as HTMLElement).getAttribute('data-checked') === 'true';
			// Trim TipTap's block wrapper; 4-space continuation keeps sublists
			// and fences nested.
			content = content.trim().replace(/\n(?=.)/g, '\n    ');
			return `- [${checked ? 'x' : ' '}] ${content}\n`;
		}
	});

	const root = document.createElement('div');
	root.innerHTML = html ?? '';
	return service.turndown(root);
};
