function i = sampDiscrete(b);

% function to sample from a (un-normalized) discrete distribution

r = sum(b)*rand();
a = b(1); i = 1;
while a < r
  i = i+1;
  a = a+b(i);
end
