struct Dropout; p; end

function (l::Dropout)(x)
    dropout(x, l.p) # TODO: dropout normalization does not depend on masks?
end

