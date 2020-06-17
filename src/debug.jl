macro size(z, s); esc(:(@assert (size($z) == $s) string(summary($z),!=,$s))); end

macro sizes(s); quote
    Ex = size(s.srcembed.w,1)
    Vx = size(s.srcembed.w,2)
    Ey = size(s.tgtembed.w,1)
    Vy = size(s.tgtembed.w,2)
    Dx = s.encoder.direction + 1
    Hx = Int(s.encoder.hiddenSize)
    Hy = Int(s.decoder.hiddenSize)
    Lx = Int(s.encoder.numLayers)
    Ly = Int(s.decoder.numLayers)
    @assert Hx == Hy && Ly == Lx*Dx
end |> esc; end