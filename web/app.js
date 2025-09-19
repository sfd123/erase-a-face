/**
 * Golf Video Anonymizer Web Interface
 * Handles file upload, validation, progress tracking, and status polling
 */

class VideoAnonymizerApp {
    constructor() {
        this.apiBaseUrl = '/api/v1';
        this.currentJobId = null;
        this.statusPollingInterval = null;
        this.selectedFile = null;
        
        // Configuration
        this.config = {
            maxFileSize: 500 * 1024 * 1024, // 500MB
            supportedFormats: ['.mp4', '.mov', '.avi'],
            pollingInterval: 2000, // 2 seconds
            maxPollingTime: 30 * 60 * 1000 // 30 minutes
        };
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadServiceConfig();
    }
    
    initializeElements() {
        // Sections
        this.uploadSection = document.getElementById('upload-section');
        this.progressSection = document.getElementById('progress-section');
        this.resultsSection = document.getElementById('results-section');
        this.errorSection = document.getElementById('error-section');
        
        // Upload elements
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.browseBtn = document.getElementById('browse-btn');
        this.fileInfo = document.getElementById('file-info');
        this.uploadBtn = document.getElementById('upload-btn');
        this.cancelBtn = document.getElementById('cancel-btn');
        
        // Progress elements
        this.progressFill = document.getElementById('progress-fill');
        this.progressText = document.getElementById('progress-text');
        this.progressPercentage = document.getElementById('progress-percentage');
        this.jobIdSpan = document.getElementById('job-id');
        this.processingFilename = document.getElementById('processing-filename');
        this.jobStatus = document.getElementById('job-status');
        
        // Results elements
        this.facesDetectedText = document.getElementById('faces-detected-text');
        this.downloadBtn = document.getElementById('download-btn');
        this.processAnotherBtn = document.getElementById('process-another-btn');
        
        // Error elements
        this.errorMessage = document.getElementById('error-message');
        this.retryBtn = document.getElementById('retry-btn');
        this.newUploadBtn = document.getElementById('new-upload-btn');
        
        // Loading overlay
        this.loadingOverlay = document.getElementById('loading-overlay');
    }
    
    attachEventListeners() {
        // File input events
        this.browseBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));
        
        // Drag and drop events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Button events
        this.uploadBtn.addEventListener('click', () => this.uploadFile());
        this.cancelBtn.addEventListener('click', () => this.resetUpload());
        this.downloadBtn.addEventListener('click', () => this.downloadVideo());
        this.processAnotherBtn.addEventListener('click', () => this.resetToUpload());
        this.retryBtn.addEventListener('click', () => this.uploadFile());
        this.newUploadBtn.addEventListener('click', () => this.resetToUpload());
        
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            document.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });
    }
    
    async loadServiceConfig() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (response.ok) {
                const config = await response.json();
                this.config.maxFileSize = config.max_file_size_mb * 1024 * 1024;
                this.config.supportedFormats = config.supported_formats.map(f => f.toLowerCase());
            }
        } catch (error) {
            console.warn('Could not load service configuration:', error);
        }
    }
    
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.handleFileSelect(files[0]);
        }
    }
    
    handleFileSelect(file) {
        if (!file) return;
        
        const validation = this.validateFile(file);
        if (!validation.valid) {
            this.showError(validation.error);
            return;
        }
        
        this.selectedFile = file;
        this.showFileInfo(file);
    }
    
    validateFile(file) {
        // Check file type
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        if (!this.config.supportedFormats.includes(fileExtension)) {
            return {
                valid: false,
                error: `Unsupported file format. Please use: ${this.config.supportedFormats.join(', ')}`
            };
        }
        
        // Check file size
        if (file.size > this.config.maxFileSize) {
            const maxSizeMB = Math.round(this.config.maxFileSize / (1024 * 1024));
            const fileSizeMB = Math.round(file.size / (1024 * 1024));
            return {
                valid: false,
                error: `File too large (${fileSizeMB}MB). Maximum size is ${maxSizeMB}MB.`
            };
        }
        
        return { valid: true };
    }
    
    showFileInfo(file) {
        const fileName = this.fileInfo.querySelector('.file-name');
        const fileSize = this.fileInfo.querySelector('.file-size');
        
        fileName.textContent = file.name;
        fileSize.textContent = this.formatFileSize(file.size);
        
        this.fileInfo.classList.remove('hidden');
        this.uploadArea.style.display = 'none';
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    async uploadFile() {
        if (!this.selectedFile) return;
        
        this.showSection('progress');
        this.updateProgress(0, 'Uploading...');
        
        const formData = new FormData();
        formData.append('file', this.selectedFile);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/upload`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.message || result.detail?.message || 'Upload failed');
            }
            
            this.currentJobId = result.job_id;
            this.jobIdSpan.textContent = result.job_id;
            this.processingFilename.textContent = result.original_filename;
            
            this.updateProgress(25, 'Processing...');
            this.startStatusPolling();
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showError(`Upload failed: ${error.message}`);
        }
    }
    
    startStatusPolling() {
        if (this.statusPollingInterval) {
            clearInterval(this.statusPollingInterval);
        }
        
        const startTime = Date.now();
        
        this.statusPollingInterval = setInterval(async () => {
            try {
                // Check for timeout
                if (Date.now() - startTime > this.config.maxPollingTime) {
                    clearInterval(this.statusPollingInterval);
                    this.showError('Processing timeout. Please try again or contact support.');
                    return;
                }
                
                const response = await fetch(`${this.apiBaseUrl}/status/${this.currentJobId}`);
                const status = await response.json();
                
                if (!response.ok) {
                    throw new Error(status.message || status.detail?.message || 'Status check failed');
                }
                
                this.updateJobStatus(status);
                
                if (status.status === 'COMPLETED') {
                    clearInterval(this.statusPollingInterval);
                    this.showResults(status);
                } else if (status.status === 'FAILED') {
                    clearInterval(this.statusPollingInterval);
                    this.showError(status.error_message || 'Processing failed');
                }
                
            } catch (error) {
                console.error('Status polling error:', error);
                // Continue polling unless it's a critical error
                if (error.message.includes('404') || error.message.includes('not found')) {
                    clearInterval(this.statusPollingInterval);
                    this.showError('Job not found. It may have been cleaned up.');
                }
            }
        }, this.config.pollingInterval);
    }
    
    updateJobStatus(status) {
        this.jobStatus.textContent = status.status;
        
        // Update progress based on status
        let progress = 25; // Base progress after upload
        let progressText = 'Processing...';
        
        switch (status.status) {
            case 'PENDING':
                progress = 30;
                progressText = 'Queued for processing...';
                break;
            case 'PROCESSING':
                progress = 60;
                progressText = 'Detecting and blurring faces...';
                break;
            case 'COMPLETED':
                progress = 100;
                progressText = 'Complete!';
                break;
            case 'FAILED':
                progress = 0;
                progressText = 'Failed';
                break;
        }
        
        this.updateProgress(progress, progressText);
    }
    
    updateProgress(percentage, text) {
        this.progressFill.style.width = `${percentage}%`;
        this.progressPercentage.textContent = `${percentage}%`;
        this.progressText.textContent = text;
    }
    
    showResults(status) {
        const facesCount = status.faces_detected || 0;
        const facesText = facesCount === 0 
            ? 'No faces were detected in your video.'
            : facesCount === 1 
                ? '1 face was detected and anonymized.'
                : `${facesCount} faces were detected and anonymized.`;
        
        this.facesDetectedText.textContent = facesText;
        this.showSection('results');
    }
    
    async downloadVideo() {
        if (!this.currentJobId) return;
        
        try {
            this.showLoading(true);
            
            const response = await fetch(`${this.apiBaseUrl}/download/${this.currentJobId}`);
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || error.detail?.message || 'Download failed');
            }
            
            // Get filename from response headers
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = 'anonymized_video.mp4';
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename=(.+)/);
                if (filenameMatch) {
                    filename = filenameMatch[1].replace(/"/g, '');
                }
            }
            
            // Create download link
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
        } catch (error) {
            console.error('Download error:', error);
            this.showError(`Download failed: ${error.message}`);
        } finally {
            this.showLoading(false);
        }
    }
    
    showSection(sectionName) {
        // Hide all sections
        [this.uploadSection, this.progressSection, this.resultsSection, this.errorSection]
            .forEach(section => section.classList.add('hidden'));
        
        // Show requested section
        switch (sectionName) {
            case 'upload':
                this.uploadSection.classList.remove('hidden');
                break;
            case 'progress':
                this.progressSection.classList.remove('hidden');
                break;
            case 'results':
                this.resultsSection.classList.remove('hidden');
                break;
            case 'error':
                this.errorSection.classList.remove('hidden');
                break;
        }
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.showSection('error');
        
        // Clear any ongoing polling
        if (this.statusPollingInterval) {
            clearInterval(this.statusPollingInterval);
            this.statusPollingInterval = null;
        }
    }
    
    showLoading(show) {
        if (show) {
            this.loadingOverlay.classList.remove('hidden');
        } else {
            this.loadingOverlay.classList.add('hidden');
        }
    }
    
    resetUpload() {
        this.selectedFile = null;
        this.fileInfo.classList.add('hidden');
        this.uploadArea.style.display = 'block';
        this.fileInput.value = '';
    }
    
    resetToUpload() {
        this.resetUpload();
        this.currentJobId = null;
        
        // Clear any ongoing polling
        if (this.statusPollingInterval) {
            clearInterval(this.statusPollingInterval);
            this.statusPollingInterval = null;
        }
        
        this.showSection('upload');
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VideoAnonymizerApp();
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VideoAnonymizerApp;
}