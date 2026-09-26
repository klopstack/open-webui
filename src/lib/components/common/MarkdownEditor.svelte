<script lang="ts">
	import { createEventDispatcher, getContext, onDestroy, onMount } from 'svelte';

	import { Editor } from '@tiptap/core';
	import StarterKit from '@tiptap/starter-kit';
	import { TableKit } from '@tiptap/extension-table';
	import { ListKit } from '@tiptap/extension-list';
	import Link from '@tiptap/extension-link';

	import { htmlToMarkdown, markdownToHtml } from '$lib/utils/markdown';

	import Modal from './Modal.svelte';
	import Tooltip from './Tooltip.svelte';
	import XMark from '../icons/XMark.svelte';
	import H1 from '../icons/H1.svelte';
	import H2 from '../icons/H2.svelte';
	import H3 from '../icons/H3.svelte';
	import Bold from '../icons/Bold.svelte';
	import Italic from '../icons/Italic.svelte';
	import Strikethrough from '../icons/Strikethrough.svelte';
	import ListBullet from '../icons/ListBullet.svelte';
	import NumberedList from '../icons/NumberedList.svelte';
	import TaskList from '../icons/TaskList.svelte';
	import Document from '../icons/Document.svelte';
	import CodeBracket from '../icons/CodeBracket.svelte';
	import Grid from '../icons/Grid.svelte';
	import LinkIcon from '../icons/Link.svelte';

	const i18n = getContext<any>('i18n');
	const eventDispatch = createEventDispatcher();

	/** The markdown source of truth (bound by the parent). */
	export let value = '';
	/** False renders a read-only editor (read-only KBs). */
	export let editable = true;
	export let className = '';

	let editor: Editor | null = null;
	let element: HTMLElement | null = null;
	/** Last markdown we exported — guards against the parent echoing our own
	 *  change back through `value` and resetting the cursor. */
	let lastExportedMd = '';

	// Link dialog state
	let showLinkModal = false;
	let linkUrl = '';

	const btnClass = (active: boolean) =>
		`${active ? 'bg-gray-50 dark:bg-gray-700' : ''} hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg p-1.5 transition-all`;

	const openLinkDialog = () => {
		linkUrl = editor?.getAttributes('link')?.href ?? '';
		showLinkModal = true;
	};

	const applyLink = () => {
		const url = linkUrl.trim();
		if (editor && url) {
			editor.chain().focus().extendMarkRange('link').setLink({ href: url }).run();
		} else if (editor) {
			editor.chain().focus().unsetLink().run();
		}
		showLinkModal = false;
	};

	const exportMarkdown = () => {
		if (!editor || editor.isDestroyed) return;
		const md = htmlToMarkdown(editor.getHTML()).trim();
		if (md !== lastExportedMd) {
			lastExportedMd = md;
			eventDispatch('change', md);
		}
	};

	onMount(() => {
		if (!element) return;
		editor = new Editor({
			element,
			extensions: [
				StarterKit.configure({
					// Link is provided by the Link extension below.
					link: false,
					// Lists come from ListKit (adds taskList/taskItem).
					bulletList: false,
					orderedList: false,
					listItem: false,
					listKeymap: false
				}),
				TableKit.configure({
					table: { resizable: false }
				}),
				ListKit.configure({
					taskItem: { nested: true }
				}),
				Link.configure({
					openOnClick: false,
					HTMLAttributes: { rel: 'noopener noreferrer', target: '_blank' }
				})
			],
			content: markdownToHtml(value),
			editable,
			onUpdate: () => {
				exportMarkdown();
			}
		});
		lastExportedMd = htmlToMarkdown(editor.getHTML()).trim();
	});

	onDestroy(() => {
		editor?.destroy();
		editor = null;
	});

	// External value changes: (a) the parent loads file.data.content
	// asynchronously after mount, or (b) the user edits the Source tab.
	// Our own exports are echoed back by the parent VERBATIM (value ===
	// lastExportedMd), so this never fires mid-keystroke in the editor and
	// never destroys the cursor. File switches remount via the parent's
	// {#key}.
	$: if (editor && !editor.isDestroyed && value !== lastExportedMd) {
		editor.commands.setContent(markdownToHtml(value), { emitUpdate: false });
		lastExportedMd = htmlToMarkdown(editor.getHTML()).trim();
	}

	$: if (editor && !editor.isDestroyed) {
		editor.setEditable(editable);
	}
</script>

<div class="md-editor flex flex-col h-full min-h-0 {className}">
	{#if editable}
		<div
			class="flex flex-wrap gap-0.5 p-0.5 rounded-xl shadow-lg bg-white text-gray-800 dark:text-white dark:bg-gray-850 min-w-fit border border-gray-100 dark:border-gray-800 mb-2 shrink-0"
		>
			<Tooltip placement="top" content="Heading 1">
				<button
					class={btnClass(!!editor?.isActive('heading', { level: 1 }))}
					type="button"
					on:click={() => editor?.chain().focus().toggleHeading({ level: 1 }).run()}
				>
					<H1 />
				</button>
			</Tooltip>
			<Tooltip placement="top" content="Heading 2">
				<button
					class={btnClass(!!editor?.isActive('heading', { level: 2 }))}
					type="button"
					on:click={() => editor?.chain().focus().toggleHeading({ level: 2 }).run()}
				>
					<H2 />
				</button>
			</Tooltip>
			<Tooltip placement="top" content="Heading 3">
				<button
					class={btnClass(!!editor?.isActive('heading', { level: 3 }))}
					type="button"
					on:click={() => editor?.chain().focus().toggleHeading({ level: 3 }).run()}
				>
					<H3 />
				</button>
			</Tooltip>

			<span class="w-px self-stretch bg-gray-200 dark:bg-gray-700 mx-0.5"></span>

			<Tooltip placement="top" content="Bold">
				<button
					class={btnClass(!!editor?.isActive('bold'))}
					type="button"
					on:click={() => editor?.chain().focus().toggleBold().run()}
				>
					<Bold />
				</button>
			</Tooltip>
			<Tooltip placement="top" content="Italic">
				<button
					class={btnClass(!!editor?.isActive('italic'))}
					type="button"
					on:click={() => editor?.chain().focus().toggleItalic().run()}
				>
					<Italic />
				</button>
			</Tooltip>
			<Tooltip placement="top" content="Strikethrough">
				<button
					class={btnClass(!!editor?.isActive('strike'))}
					type="button"
					on:click={() => editor?.chain().focus().toggleStrike().run()}
				>
					<Strikethrough />
				</button>
			</Tooltip>

			<span class="w-px self-stretch bg-gray-200 dark:bg-gray-700 mx-0.5"></span>

			<Tooltip placement="top" content="Bullet list">
				<button
					class={btnClass(!!editor?.isActive('bulletList'))}
					type="button"
					on:click={() => editor?.chain().focus().toggleBulletList().run()}
				>
					<ListBullet />
				</button>
			</Tooltip>
			<Tooltip placement="top" content="Numbered list">
				<button
					class={btnClass(!!editor?.isActive('orderedList'))}
					type="button"
					on:click={() => editor?.chain().focus().toggleOrderedList().run()}
				>
					<NumberedList />
				</button>
			</Tooltip>
			<Tooltip placement="top" content="Task list">
				<button
					class={btnClass(!!editor?.isActive('taskList'))}
					type="button"
					on:click={() => editor?.chain().focus().toggleTaskList().run()}
				>
					<TaskList />
				</button>
			</Tooltip>

			<span class="w-px self-stretch bg-gray-200 dark:bg-gray-700 mx-0.5"></span>

			<Tooltip placement="top" content="Quote">
				<button
					class={btnClass(!!editor?.isActive('blockquote'))}
					type="button"
					on:click={() => editor?.chain().focus().toggleBlockquote().run()}
				>
					<Document />
				</button>
			</Tooltip>
			<Tooltip placement="top" content="Code block">
				<button
					class={btnClass(!!editor?.isActive('codeBlock'))}
					type="button"
					on:click={() => editor?.chain().focus().toggleCodeBlock().run()}
				>
					<CodeBracket />
				</button>
			</Tooltip>
			<Tooltip placement="top" content="Table">
				<button
					class={btnClass(!!editor?.isActive('table'))}
					type="button"
					on:click={() => editor?.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()}
				>
					<Grid />
				</button>
			</Tooltip>
			<Tooltip placement="top" content="Link">
				<button
					class={btnClass(!!editor?.isActive('link'))}
					type="button"
					on:click={openLinkDialog}
				>
					<LinkIcon />
				</button>
			</Tooltip>
		</div>
	{/if}

	<div class="flex-1 min-h-0 overflow-y-auto scrollbar-hidden">
		<div class="prose dark:prose-invert max-w-full" bind:this={element}></div>
	</div>
</div>

<Modal bind:show={showLinkModal} size="sm">
	<div class="flex flex-col gap-3">
		<div class="flex items-center justify-between">
			<div class="font-medium text-sm">{$i18n.t('Link')}</div>
			<button on:click={() => (showLinkModal = false)}>
				<XMark />
			</button>
		</div>
		<input
			type="url"
			class="w-full text-sm rounded-xl bg-gray-50 dark:bg-gray-850 px-3 py-2 outline-hidden"
			placeholder="https://"
			bind:value={linkUrl}
			on:keydown={(e) => {
				if (e.key === 'Enter') applyLink();
			}}
		/>
		<div class="flex justify-end gap-2">
			<button
				class="px-3 py-1.5 rounded-xl text-xs bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition"
				on:click={() => (showLinkModal = false)}
			>
				{$i18n.t('Cancel')}
			</button>
			<button
				class="px-3 py-1.5 rounded-xl text-xs bg-black text-white dark:bg-white dark:text-black hover:opacity-90 transition"
				on:click={applyLink}
			>
				{$i18n.t('Apply')}
			</button>
		</div>
	</div>
</Modal>

<style>
	/* Scoped to .md-editor so the global .tiptap styles of the chat/note
	 * editors are untouched. */
	.md-editor :global(.tiptap) {
		min-height: 100%;
		height: 100%;
		outline: none;
		font-size: 0.875rem;
		line-height: 1.5rem;
	}

	.md-editor :global(.tiptap p.is-editor-empty:first-child)::before {
		content: attr(data-placeholder);
		float: left;
		color: rgb(0 0 0 / 0.4);
		pointer-events: none;
		height: 0;
	}

	.dark .md-editor :global(.tiptap p.is-editor-empty:first-child)::before {
		color: rgb(255 255 255 / 0.4);
	}

	.md-editor :global(.tiptap table) {
		width: 100%;
		border-collapse: collapse;
		margin: 0.75rem 0;
		font-size: 0.8125rem;
	}

	.md-editor :global(.tiptap th),
	.md-editor :global(.tiptap td) {
		border: 1px solid var(--color-gray-300, #cdcdcd);
		padding: 0.375rem 0.625rem;
		vertical-align: top;
		min-width: 2.5rem;
	}

	.dark .md-editor :global(.tiptap th),
	.dark .md-editor :global(.tiptap td) {
		border-color: var(--color-gray-600, #676767);
	}

	.md-editor :global(.tiptap th) {
		background: var(--color-gray-100, #ececec);
		font-weight: 600;
		text-align: left;
	}

	.dark .md-editor :global(.tiptap th) {
		background: var(--color-gray-800, #333);
	}

	.md-editor :global(.tiptap ul[data-type='taskList']) {
		list-style: none;
		padding-left: 0.25rem;
	}

	.md-editor :global(.tiptap ul[data-type='taskList'] li) {
		display: flex;
		align-items: flex-start;
		gap: 0.5rem;
	}

	.md-editor :global(.tiptap ul[data-type='taskList'] li > label) {
		flex: 0 0 auto;
		margin-top: 0.2rem;
		user-select: none;
	}

	.md-editor :global(.tiptap ul[data-type='taskList'] li > div) {
		flex: 1 1 auto;
	}

	.md-editor :global(.tiptap pre) {
		background: var(--color-gray-100, #ececec);
		border-radius: 0.5rem;
		font-size: 0.8125rem;
	}

	.dark .md-editor :global(.tiptap pre) {
		background: var(--color-gray-800, #333);
	}

	.md-editor :global(.tiptap code) {
		font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
		font-size: 0.8125rem;
	}

	.md-editor :global(.tiptap:not(pre) > code) {
		background: var(--color-gray-100, #ececec);
		border-radius: 0.25rem;
		padding: 0.1rem 0.3rem;
	}

	.dark .md-editor :global(.tiptap:not(pre) > code) {
		background: var(--color-gray-800, #333);
	}
</style>
