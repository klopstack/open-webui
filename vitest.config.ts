import { defineConfig } from 'vitest/config';

// Standalone vitest config (kept apart from vite.config.ts so the app build
// is untouched). Unit tests for $lib/utils run in happy-dom. Component tests
// can adopt the sveltekit plugin here later if needed.
export default defineConfig({
	test: {
		environment: 'happy-dom',
		include: ['src/**/*.spec.ts', 'src/**/*.test.ts'],
		passWithNoTests: true
	}
});
