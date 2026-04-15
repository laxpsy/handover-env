# handover-env 
raylib based RL env for simulation of cellular handovers. 

## setup for dev
1. clone repo locally
2. run `uv install`
3. source the venv (if needed)

## todo
- [ ] basic mvp (logical)
    - [x] init
    - [x] define all required entities
    - [x] implement helper functions for updating handovers/calculating rsrp and so on
    - [x] implement basic architecture
        - [x] `HandoverPolicy` 
        - [x] `SimulationConfig` 
        - [ ] `Simulation`
            - [ ] metrics collection
            - [ ] step functionality 
    - [ ] implement naive 3gpp handover algorithm w TTT support
    - [ ] ue random motion model
- [ ] basic mvp (rendering)
    - [ ] init 
    - [ ] topology/placement configuration
    - [ ] bss rendering/ue rendering
    - [ ] handover visualization 
- [ ] post-mvp footures 
    - [ ] gymnasium compatibility
    - [ ] 5g/4g bss classification
    - [ ] graphs for statistics
