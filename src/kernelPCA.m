function [ projectedPoints, eigenVectors, diagonal ] = kernelPCA( gramMatrix )

[nDataPoints,dummy] = size(gramMatrix);
assert( nDataPoints == dummy );

oneMatrix = ones(size(gramMatrix))./nDataPoints;
transformedMatrix = gramMatrix - oneMatrix*gramMatrix - gramMatrix*oneMatrix + oneMatrix*gramMatrix*oneMatrix;
transformedMatrix = 0.5*(transformedMatrix+transformedMatrix');
[eigenVectors,D] = eig( transformedMatrix );
diagonal = diag(D);

[B,IX] = sort(diagonal,'descend');
for i = 1:size(eigenVectors,2)
    EV(:,i) = eigenVectors(:,IX(i));
end

eigenVectors = EV;
diagonal = B;

projectedPoints = eigenVectors'*gramMatrix;


end

