module AxisArrayHelpers
import AxisArrayConversion
const AC = AxisArrayConversion
using Requires

include("interpolations.jl")

function __init__()
    @require CUDA= "052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

end
