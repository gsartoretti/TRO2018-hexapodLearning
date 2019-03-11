classdef Controller
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    % Channels
    
    % 2         4
    % ^         ^ 
    % |         |
    % L _ > 1   L _ > 3
    
    properties
        
        joy
        
        dir
        
        ch
        deadband
        
        pressed
        lastPressed
        
    end
    
    methods
        
        %Setup
        function this = Controller(joy)
            
            this.joy = vrjoystick(joy);
            
            this.dir = -1;
            
            this.ch = zeros(1,13);
            this.deadband = 0.35;
            
            this.pressed = zeros(1,13);
            this.lastPressed = zeros(1,13);
            
        end
        
        
        %Update
        function this = updateController(this)
            
            this.lastPressed = this.pressed;
            [this.ch, this.pressed, this.dir] = read(this.joy);
            
            this.ch = this.ch .* (abs(this.ch) > this.deadband) .* [1 -1 1 -1];
            
        end
        
        %Setters
        
        function this = setDeadband(this,deadband)
            this.deadband = deadband;
        end
        
        %Getters
        
        function state = dirPad(this)
            state = this.dir;   
        end
        
        function state = channel(this,num)
            state = this.ch(num);
        end
        
        function state = button(this,num)
            state = this.pressed(num);
        end
        
        function state = buttonPressed(this,num)
            state = this.pressed(num) & ~this.lastPressed(num);
        end
        
        %Tools
        
        function value = adjustValue(this,up,down,value,dV)
            if this.buttonPressed(up)
                value = value + dV;
            elseif this.buttonPressed(down)
                value = value - dV;
            end
        end
            
        
    end
    
end

