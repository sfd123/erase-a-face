#!/usr/bin/env node

/**
 * Test runner for Golf Video Anonymizer Frontend Tests
 * Usage: node web/run-tests.js
 */

const fs = require('fs');
const path = require('path');

// Check if we're in the right directory
const testFile = path.join(__dirname, 'app.test.js');
if (!fs.existsSync(testFile)) {
    console.error('❌ Test file not found. Make sure you\'re running this from the project root.');
    process.exit(1);
}

console.log('🧪 Golf Video Anonymizer Frontend Test Runner\n');

try {
    // Load and run tests
    require('./app.test.js');
    
    console.log('\n✅ Test execution completed successfully');
    
} catch (error) {
    console.error('\n❌ Test execution failed:', error.message);
    process.exit(1);
}