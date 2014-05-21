function SetDefaultValue(position, argName, defaultValue)
% Author: Richie Cotton
if evalin('caller', 'nargin') < position || ...
      isempty(evalin('caller', argName))
   assignin('caller', argName, defaultValue);
end
end
