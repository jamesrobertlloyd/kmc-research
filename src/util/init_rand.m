function init_rand( seed )
%INITIALISERAND Seeds rand and randn
% Not really sure about differences between twister and state
% If you know, please comment / change
%
% James Lloyd, June 2012
  rand('twister', seed);
  randn('state', seed);
end

