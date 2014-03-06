function [ projectedPoints, eigenVectors, diagonal ] = kernelPCA4( gramMatrix )

[nDataPoints,dummy] = size(gramMatrix);
assert( nDataPoints == dummy );

oneMatrix = ones(size(gramMatrix))./nDataPoints;
transformedMatrix = gramMatrix - oneMatrix*gramMatrix - gramMatrix*oneMatrix + oneMatrix*gramMatrix*oneMatrix;
transformedMatrix = 0.5*(transformedMatrix+transformedMatrix');
[eigenVectors,D] = eig( transformedMatrix );
diagonal = diag(D)./nDataPoints;

[B,IX] = sort(diagonal,'descend');
for i = 1:size(eigenVectors,2)
    EV(:,i) = eigenVectors(:,IX(i))./sqrt(nDataPoints*B(i));
end

eigenVectors = EV;
diagonal = B;

projectedPoints = eigenVectors'*transformedMatrix;


end

