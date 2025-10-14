/**
 * Phase 0 Test: Verify frontend project structure
 */

import { existsSync } from 'fs';
import { join } from 'path';

describe('Frontend Project Structure', () => {
  const frontendPath = join(process.cwd(), 'frontend');
  
  test('should have cleaned up module references', () => {
    // Check that allocation_input_form.jsx exists (renamed from order_input_form.jsx)
    expect(existsSync(join(frontendPath, 'allocation_input_form.jsx'))).toBe(true);
    
    // Check that old order_input_form.jsx doesn't exist
    expect(existsSync(join(frontendPath, 'order_input_form.jsx'))).toBe(false);
  });

  test('should have Module 4 files', () => {
    expect(existsSync(join(frontendPath, 'module4.jsx'))).toBe(true);
    expect(existsSync(join(frontendPath, 'module4_planning.jsx'))).toBe(true);
    expect(existsSync(join(frontendPath, 'App.tsx'))).toBe(true);
  });

  test('should have required component files', () => {
    expect(existsSync(join(frontendPath, 'allocation_input_form.jsx'))).toBe(true);
    expect(existsSync(join(frontendPath, 'allocation_results.jsx'))).toBe(true);
    expect(existsSync(join(frontendPath, 'decision_factors_table.jsx'))).toBe(true);
    expect(existsSync(join(frontendPath, 'planning_table.jsx'))).toBe(true);
  });

  test('should have package.json with correct project info', () => {
    const packageJsonPath = join(frontendPath, 'package.json');
    expect(existsSync(packageJsonPath)).toBe(true);
    
    const packageJson = require(packageJsonPath);
    expect(packageJson.name).toBe('allocation-maximizer');
    expect(packageJson.description).toContain('Module 4');
    expect(packageJson.description).toContain('Allocation Maximizer');
  });
});