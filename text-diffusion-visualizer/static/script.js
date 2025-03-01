// script.js
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const promptInput = document.getElementById('prompt-input');
    const generateBtn = document.getElementById('generate-btn');
    const loadingIndicator = document.getElementById('loading');
    const errorMessage = document.getElementById('error-message');
    const visualizationContainer = document.getElementById('visualization-container');
    const tokenGrid = document.getElementById('token-grid');
    const playPauseBtn = document.getElementById('play-pause');
    const stepSlider = document.getElementById('step-slider');
    const currentStepDisplay = document.getElementById('current-step');
    const totalStepsDisplay = document.getElementById('total-steps');
    const outputSection = document.getElementById('output-section');
    const outputText = document.getElementById('output-text');
    const genLengthDisplay = document.getElementById('gen-length-display');
    const stepsDisplay = document.getElementById('steps-display');
    const blockLengthDisplay = document.getElementById('block-length-display');

    // Visualization state
    let states = [];
    let currentStep = 0;
    let isPlaying = false;
    let animationId = null;
    let maskId = 126336;  // [MASK] token ID
    let promptLength = 0;

    // Initialize the UI
    function resetUI() {
        errorMessage.classList.add('hidden');
        errorMessage.textContent = '';
        visualizationContainer.classList.add('hidden');
        outputSection.classList.add('hidden');
        tokenGrid.innerHTML = '';
        currentStep = 0;
        states = [];
        isPlaying = false;
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
        playPauseBtn.textContent = 'Play';
    }

    // Handle generation
    generateBtn.addEventListener('click', function() {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            showError('Please enter a prompt first.');
            return;
        }

        resetUI();
        loadingIndicator.classList.remove('hidden');
        generateBtn.disabled = true;

        fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: prompt })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to generate response');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            loadingIndicator.classList.add('hidden');
            generateBtn.disabled = false;

            // Update configuration display
            genLengthDisplay.textContent = data.config.gen_length;
            stepsDisplay.textContent = data.config.steps;
            blockLengthDisplay.textContent = data.config.block_length;

            // Update state data
            states = data.states;
            promptLength = data.prompt_length;
            
            // Update UI
            visualizationContainer.classList.remove('hidden');
            outputSection.classList.remove('hidden');
            outputText.textContent = data.answer;
            
            // Set up slider
            totalStepsDisplay.textContent = states.length - 1;
            stepSlider.max = states.length - 1;
            stepSlider.value = 0;
            stepSlider.disabled = false;
            
            // Render initial state
            renderState(0);
        })
        .catch(error => {
            loadingIndicator.classList.add('hidden');
            generateBtn.disabled = false;
            showError(error.message || 'An error occurred during generation.');
        });
    });

    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    }

    // Render a specific state
    function renderState(stateIndex) {
        if (!states || !states[stateIndex]) {
            return;
        }

        currentStep = stateIndex;
        currentStepDisplay.textContent = stateIndex;
        stepSlider.value = stateIndex;
        
        // Clear previous tokens
        tokenGrid.innerHTML = '';
        
        // Get current state
        const state = states[stateIndex];
        
        // Render each token
        state.forEach((token, index) => {
            const tokenElement = document.createElement('div');
            tokenElement.classList.add('token');
            tokenElement.classList.add(token.type);
            
            // Use first character for display or '?' for mask
            const displayText = token.id === maskId ? '?' : 
                token.text.replace(/\s/g, '·').slice(0, 2);
            tokenElement.textContent = displayText;
            
            // Add tooltip
            const tooltipElement = document.createElement('span');
            tooltipElement.classList.add('token-info');
            
            // For tooltip, show actual token text or "[MASK]" 
            tooltipElement.textContent = token.id === maskId ? '[MASK]' : 
                token.text.replace(/\s/g, '·');
            
            tokenElement.appendChild(tooltipElement);
            tokenGrid.appendChild(tokenElement);
        });
    }

    // Play/Pause the animation
    playPauseBtn.addEventListener('click', function() {
        if (!states || states.length === 0) {
            return;
        }
        
        isPlaying = !isPlaying;
        playPauseBtn.textContent = isPlaying ? 'Pause' : 'Play';
        
        if (isPlaying) {
            playAnimation();
        } else if (animationId) {
            cancelAnimationFrame(animationId);
        }
    });

    // Handle slider changes
    stepSlider.addEventListener('input', function() {
        if (!states || states.length === 0) {
            return;
        }
        
        // Pause any ongoing animation
        isPlaying = false;
        playPauseBtn.textContent = 'Play';
        if (animationId) {
            cancelAnimationFrame(animationId);
        }
        
        // Render the selected state
        renderState(parseInt(stepSlider.value));
    });

    // Animation function
    function playAnimation() {
        if (!isPlaying) {
            return;
        }
        
        // If we've reached the end, stop playing
        if (currentStep >= states.length - 1) {
            isPlaying = false;
            playPauseBtn.textContent = 'Play';
            return;
        }
        
        // Advance to next step
        renderState(currentStep + 1);
        
        // Schedule next frame with a delay for visibility
        animationId = requestAnimationFrame(() => {
            setTimeout(playAnimation, 100); // Control animation speed (100ms delay)
        });
    }

    // Allow Enter key in text input to trigger generation
    promptInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !generateBtn.disabled) {
            generateBtn.click();
        }
    });

    // Add example prompt
    promptInput.value = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?";
});