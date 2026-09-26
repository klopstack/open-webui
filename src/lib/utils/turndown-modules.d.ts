/**
 * Ambient declarations for the untyped turndown packages (neither ships
 * types, and no @types/* package exists for the joplin fork). Kept minimal —
 * only the surface used by $lib/utils/markdown.ts.
 */
declare module 'turndown' {
	export interface TurndownRule {
		filter: string | string[] | ((node: HTMLElement, options: unknown) => boolean);
		replacement: (content: string, node: HTMLElement, options: unknown) => string;
	}

	export interface TurndownOptions {
		root?: HTMLElement;
		headingStyle?: 'setext' | 'atx';
		horizontalRule?: string;
		bulletListMarker?: string;
		codeBlockStyle?: 'fenced' | 'indented';
		fence?: string;
		emoji?: 'decimal' | 'unicode' | 'short';
		strongDelimiter?: string;
		emDelimiter?: string;
		linkStyle?: 'inlined' | 'referenced';
		linkReferenceStyle?: 'full' | 'collapsed' | 'shortcut';
		paragraphSeparator?: string;
	}

	export default class TurndownService {
		constructor(options?: TurndownOptions);
		rules: {
			add: (key: string, rule: TurndownRule) => void;
			keep: (filter: TurndownRule['filter']) => void;
			remove: (filter: TurndownRule['filter']) => void;
		};
		use: (plugin: unknown | unknown[]) => this;
		addRule: (key: string, rule: TurndownRule) => this;
		keep: (filter: TurndownRule['filter']) => this;
		remove: (filter: TurndownRule['filter']) => this;
		escape: (string: string) => string;
		isCodeBlock: (node: HTMLElement) => boolean;
		options: TurndownOptions;
		turndown: (input: string | HTMLElement) => string;
	}
}

declare module '@joplin/turndown-plugin-gfm' {
	interface ServiceLike {
		use: (plugin: unknown | unknown[]) => unknown;
		addRule: (key: string, rule: unknown) => unknown;
		keep: (filter: unknown) => unknown;
		isCodeBlock: (node: HTMLElement) => boolean;
		options: unknown;
	}
	export function gfm(service: ServiceLike): void;
	export function highlightedCodeBlock(service: ServiceLike): void;
	export function strikethrough(service: ServiceLike): void;
	export function tables(service: ServiceLike): void;
	export function taskListItems(service: ServiceLike): void;
}
