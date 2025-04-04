include("MOI_wrapper.jl")
power_file = ENV["PGLIB"] * "/pglib_opf_case3_lmbd.m";
function make_test_model(data, optimizer)
    Y = calc_admittance_matrix(data).matrix

    n = Y.m;
    C = zeros(n,n)
    q = zeros(n)
    C[1,1] = data["gen"]["1"]["cost"][1]
    C[2,2] = data["gen"]["2"]["cost"][1]
    I = diagm(ones(2n))
    q[1] = data["gen"]["1"]["cost"][2]
    q[2] = data["gen"]["2"]["cost"][2]
    test_model = Model(optimizer)
    @variable(test_model, U[1:2N])
    @variable(test_model, P[1:N])
    @variable(test_model, Q[1:N])
    @objective(test_model, MIN_SENSE, P' * C * P + q' * P)
    @constraint(test_model, U[4]==0)
    for j in 1:N
        eⱼ = zeros(N); eⱼ[j] = 1; Eⱼ = diagm(eⱼ);Ψⱼ = to_real_rep(Eⱼ * Y);Φⱼ = to_real_rep(-im * Eⱼ * Y);
        @constraint(test_model, U' * Ψⱼ * U - P[j] <= -data["load"]["$(j)"]["pd"]) 
        @constraint(test_model, U' * Φⱼ * U - Q[j] <= -data["load"]["$(j)"]["qd"]) 
        @constraint(test_model, data["gen"]["$(j)"]["pmin"] <= P[j] <= data["gen"]["$(j)"]["pmax"])
        @constraint(test_model, data["gen"]["$(j)"]["qmin"] <= Q[j] <= data["gen"]["$(j)"]["qmax"])
        @constraint(test_model, 0.81 <= U[j]^2 + U[j+N]^2 <= 1.21)
        # @constraint(test_model, U[j]^2 + U[j+N]^2 >= 0.9)
    end
    return test_model
end
data = parse_file(power_file)
# for (_, bdict) in data["branch"]
#     delete!(bdict, "rate_a")
#     delete!(bdict, "rate_b")
#     delete!(bdict, "rate_c")
# end

pm = instantiate_model(data, ACRPowerModel, PowerModels.build_opf);
model = pm.model.moi_backend
N = 3
# set_silent(pm.model)
result = optimize_model!(pm, optimizer=Ipopt.Optimizer)
V = vcat([result["solution"]["bus"]["$(i)"]["vr"] for i in 1:3], [result["solution"]["bus"]["$(i)"]["vi"] for i in 1:3])
P = zeros(2N)#vcat([result["solution"]["gen"]["$(i)"]["pg"] for i in 1:3], [result["solution"]["gen"]["$(i)"]["qg"] for i in 1:3])
Pb = []
for (i, (b, bdict)) in enumerate(result["solution"]["branch"])
    bval = data["branch"][string( i)]
    src = convert(Int, bval["f_bus"])
    dst = convert(Int, bval["t_bus"])
    pflow_from = bdict["pf"]
    pflow_to = bdict["pt"]
    P[src] += pflow_from
    P[dst] += pflow_to
    qflow_from = bdict["qf"]
    qflow_to = bdict["qt"]
    P[src+N] += qflow_from
    P[dst+N] += qflow_to
end