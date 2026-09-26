<script lang="ts">
	import dayjs from '$lib/dayjs';
	import duration from 'dayjs/plugin/duration';
	import relativeTime from 'dayjs/plugin/relativeTime';

	dayjs.extend(duration);
	dayjs.extend(relativeTime);

	import { getContext } from 'svelte';
	const i18n = getContext('i18n');

	import { capitalizeFirstLetter, formatFileSize } from '$lib/utils';
	import { WEBUI_BASE_URL } from '$lib/constants';

	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Dropdown from '$lib/components/common/Dropdown.svelte';
	import DropdownMenu from '$lib/components/common/DropdownMenu.svelte';
	import DocumentPage from '$lib/components/icons/DocumentPage.svelte';
	import EllipsisHorizontal from '$lib/components/icons/EllipsisHorizontal.svelte';
	import Download from '$lib/components/icons/Download.svelte';
	import GarbageBin from '$lib/components/icons/GarbageBin.svelte';
	import Pencil from '$lib/components/icons/Pencil.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import DirectoryRow from './DirectoryRow.svelte';

	// Consolidated document metadata — written by the docbrowser plugin into
	// the KB file's upload metadata (meta.data.doc_meta). See the owui-docs
	// repo for the full contract; the fields below are what this UI consumes.
	type DocMeta = {
		schema?: number;
		doc_key?: string;
		title?: string;
		category?: string;
		media_type?: string;
		primary?: {
			role?: string;
			file_id?: string | null;
			relpath?: string;
			media_type?: string;
			extent?: { unit?: string; value?: number } | null;
		};
		markdown?: { file_id?: string; relpath?: string };
		artifacts?: {
			role?: string;
			file_id?: string | null;
			relpath?: string;
			media_type?: string;
		}[];
		status?: string;
		flags?: string[];
	};

	type KnowledgeFile = {
		id?: string;
		tempId?: string;
		itemId?: string;
		name?: string;
		status?: string;
		meta?: {
			name?: string;
			size?: number;
			data?: {
				doc_meta?: DocMeta;
				external_ref?: { path?: string };
			};
		};
		updated_at?: number;
		user?: {
			email?: string;
			name?: string;
		};
	};

	export let knowledge = null;
	export let selectedFileId = null;
	export let files: KnowledgeFile[] = [];
	export let directories = [];

	export let onClick: (fileId: string | undefined) => void = () => {};
	export let onDelete: (fileId: string | undefined) => void = () => {};
	export let onRename: (fileId: string, name: string) => void = () => {};
	export let onNavigateDirectory: (directoryId: string) => void = () => {};
	export let onRenameDirectory: (id: string, name: string) => void = () => {};
	export let onDeleteDirectory: (id: string) => void = () => {};
	export let onMoveFileToDirectory: (fileId: string, directoryId: string) => void = () => {};
	export let onMoveDirectoryToDirectory: (
		dirId: string,
		targetDirectoryId: string
	) => void = () => {};

	let editingFileId: string | null = null;
	let editName = '';
	let editInput: HTMLInputElement;

	const startRename = (file: KnowledgeFile) => {
		editingFileId = file?.id ?? file?.tempId;
		editName = file?.name ?? file?.meta?.name ?? '';
		setTimeout(() => editInput?.select(), 0);
	};

	const submitRename = () => {
		if (editingFileId && editName.trim()) {
			onRename(editingFileId, editName.trim());
		}
		editingFileId = null;
	};

	const cancelRename = () => {
		editingFileId = null;
	};

	// ── Consolidated document groups (docbrowser plugin contract) ─────────────
	// One row per DOCUMENT: files sharing a doc_key (from meta.data.doc_meta,
	// falling back to the by-date stem of a reference path) collapse into a
	// single meta-entry. Files without a doc_key render as before (one row
	// each) — manual KBs and pre-migration layouts degrade gracefully.
	const ROLE_SUFFIXES = ['_orig_front', '_orig_back', '_collated', '_layered', '_audio', '_orig'];

	const docKeyFor = (file: KnowledgeFile): string | null => {
		const dm = file?.meta?.data?.doc_meta;
		if (dm?.doc_key) return dm.doc_key;
		const ref = file?.meta?.data?.external_ref?.path;
		if (typeof ref === 'string' && ref.startsWith('ref:by-date/')) {
			const base = ref.split('/').pop() ?? '';
			const stem = base.includes('.') ? base.slice(0, base.lastIndexOf('.')) : base;
			for (const suf of ROLE_SUFFIXES) {
				if (stem.endsWith(suf)) return stem.slice(0, -suf.length);
			}
			return stem || null;
		}
		return null;
	};

	const extentLabel = (dm: DocMeta | null | undefined): string | null => {
		const ex = dm?.primary?.extent;
		if (!ex || typeof ex.value !== 'number') return null;
		return ex.unit ? `${ex.value} ${ex.unit}` : String(ex.value);
	};

	type FileGroup = {
		key: string;
		rep: KnowledgeFile; // representative row (markdown / doc_meta holder)
		members: KnowledgeFile[];
		docMeta: DocMeta | null;
	};

	const isMarkdownRow = (f: KnowledgeFile) =>
		!!f?.meta?.data?.doc_meta || (f?.name ?? '').toLowerCase().endsWith('.md');

	$: groups = (() => {
		const map = new Map<string, FileGroup>();
		for (const file of files) {
			const dk = docKeyFor(file);
			const gk = dk ? `doc:${dk}` : `solo:${file?.id ?? file?.tempId ?? file?.itemId}`;
			const existing = map.get(gk);
			if (!existing) {
				map.set(gk, {
					key: gk,
					rep: file,
					members: [file],
					docMeta: file?.meta?.data?.doc_meta ?? null,
				});
			} else {
				existing.members.push(file);
				if (isMarkdownRow(file) && !isMarkdownRow(existing.rep)) existing.rep = file;
				if (!existing.docMeta && file?.meta?.data?.doc_meta)
					existing.docMeta = file.meta.data.doc_meta;
			}
		}
		return [...map.values()];
	})();
</script>

<div class=" max-h-full flex flex-col w-full gap-[0.03125rem]" role="list">
	<!-- Directories first -->
	{#each directories as dir (dir.id)}
		<DirectoryRow
			directory={dir}
			writeAccess={knowledge?.write_access}
			onNavigate={(id) => onNavigateDirectory(id)}
			onRename={(id, name) => onRenameDirectory(id, name)}
			onDelete={(id) => onDeleteDirectory(id)}
			onFileDrop={(fileId, directoryId) => onMoveFileToDirectory(fileId, directoryId)}
			onDirDrop={(dirId, targetId) => onMoveDirectoryToDirectory(dirId, targetId)}
		/>
	{/each}

	<!-- Files (one row per document group; solo files render as before) -->
	{#each groups as group (group.key)}
		{@const file = group.rep}
		{@const docMeta = group.docMeta}
		{@const extent = extentLabel(docMeta)}
		{@const subDocs = group.members.length}
		<div
			class=" flex cursor-pointer w-full px-2 bg-transparent dark:hover:bg-gray-850/50 hover:bg-white rounded-xl transition {selectedFileId
				? ''
				: 'hover:bg-gray-100 dark:hover:bg-gray-850'}"
			role="listitem"
			draggable="true"
			on:dragstart={(e) => {
				const fileId = file?.id ?? file?.tempId;
				if (fileId) {
					e.dataTransfer?.setData('application/x-kb-file-move', JSON.stringify({ fileId }));
				}
			}}
		>
			<div class="flex items-center">
				{#if file?.status !== 'uploading'}
					<button
						class="p-1 rounded-full transition"
						type="button"
						on:click={() => {
							onClick(file?.id ?? file?.tempId);
						}}
					>
						<DocumentPage className="size-3.5" />
					</button>
				{:else}
					<Spinner className="size-3.5" />
				{/if}
			</div>

			<button
				class="relative flex items-center gap-1 rounded-xl p-2 text-left flex-1 justify-between"
				type="button"
				on:click={() => {
					if (editingFileId) return;
					onClick(file?.id ?? file?.tempId);
				}}
				on:dblclick={() => {
					if (knowledge?.write_access) startRename(file);
				}}
			>
				<div>
					<div class="flex gap-2 items-center line-clamp-1">
						{#if editingFileId === (file?.id ?? file?.tempId)}
							<!-- svelte-ignore a11y-autofocus -->
							<input
								bind:this={editInput}
								bind:value={editName}
								class="text-xs w-full bg-transparent border-none outline-hidden"
								on:keydown={(e) => {
									if (e.key === 'Enter') submitRename();
									if (e.key === 'Escape') cancelRename();
									if (e.key === ' ') e.stopPropagation();
								}}
								on:keyup={(e) => {
									if (e.key === ' ') e.stopPropagation();
								}}
								on:blur={submitRename}
								on:click={(e) => e.stopPropagation()}
								autofocus
							/>
						{:else}
							<div class="line-clamp-1 text-xs">
								{docMeta?.title ?? file?.name ?? file?.meta?.name}
								{#if subDocs > 1}
									<span
										class="rounded-md bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 text-[0.625rem] leading-none text-gray-500 dark:text-gray-400"
										title={$i18n.t('{{count}} sub-documents', { count: subDocs })}
									>{$i18n.t('{{count}} sub-documents', { count: subDocs })}</span
									>
								{/if}
								{#if file?.meta?.size}
									<span class="text-[0.6875rem] text-gray-500"
										>{formatFileSize(file?.meta?.size)}</span
									>
								{/if}
							</div>
						{/if}
					</div>
				</div>

				<div class="flex items-center gap-2 shrink-0">
					{#if extent}
						<span
							class="rounded-md bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 text-[0.625rem] leading-none text-gray-500 dark:text-gray-400"
							title={docMeta?.primary?.relpath ?? ''}
							>{extent}</span
						>
					{/if}
				{#if file?.updated_at}
					<Tooltip content={dayjs(file.updated_at * 1000).format('LLLL')}>
						<div class="text-xs text-gray-400">
							{dayjs(file.updated_at * 1000).fromNow()}
						</div>
					</Tooltip>
				{/if}

					{#if file?.user}
						<Tooltip
							content={file?.user?.email ?? $i18n.t('Deleted User')}
							className="flex shrink-0"
							placement="top-start"
						>
							<div class="shrink-0 text-gray-500">
								{$i18n.t('By {{name}}', {
									name: capitalizeFirstLetter(
										file?.user?.name ?? file?.user?.email ?? $i18n.t('Deleted User')
									)
								})}
							</div>
						</Tooltip>
					{/if}
				</div>
			</button>

			{#if knowledge?.write_access}
				<div class="flex items-center">
					<Dropdown align="end" sideOffset={4}>
						<button
							class="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-850 transition"
							type="button"
						>
							<EllipsisHorizontal className="size-3.5" />
						</button>

						<div slot="content">
							<DropdownMenu className="min-w-[8.75rem] z-[9999999]">
								<button
									type="button"
									class="select-none flex h-[1.6875rem] w-full cursor-pointer items-center gap-2 rounded-xl bg-transparent px-2 text-xs transition hover:text-gray-900 dark:hover:text-gray-100"
									on:click={() => {
										startRename(file);
									}}
								>
									<Pencil className="size-3.5" />
									{$i18n.t('Rename')}
								</button>
								<button
									type="button"
									class="select-none flex h-[1.6875rem] w-full cursor-pointer items-center gap-2 rounded-xl bg-transparent px-2 text-xs transition hover:text-gray-900 dark:hover:text-gray-100"
									on:click={() => {
									// Consolidated docs download the PRIMARY artifact
									// (layered PDF / audio / original) via its reference
									// row; plain files download themselves.
									const fileId = docMeta?.primary?.file_id ?? file?.id ?? file?.tempId;
									if (fileId) {
										window.open(`${WEBUI_BASE_URL}/api/v1/files/${fileId}/content`, '_blank');
									}
									}}
								>
									<Download className="size-3.5" />
									{$i18n.t('Download')}
								</button>
								<button
									type="button"
									class="select-none flex h-[1.6875rem] w-full cursor-pointer items-center gap-2 rounded-xl bg-transparent px-2 text-xs transition hover:text-gray-900 dark:hover:text-gray-100"
									on:click={() => {
										onDelete(file?.id ?? file?.tempId);
									}}
								>
									<GarbageBin className="size-3.5" />
									{$i18n.t('Delete')}
								</button>
							</DropdownMenu>
						</div>
					</Dropdown>
				</div>
			{/if}
		</div>
	{/each}
</div>
