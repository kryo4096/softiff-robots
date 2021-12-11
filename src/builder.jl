using GLMakie
using StaticArrays

module SimulationBuilder

    using GLMakie
    using LinearAlgebra
    using Gtk
    using JSON

    const I = Int64
    const S = Float64

    @enum PlacementMode SelectMode VertexMode TriangleMode ActuatorMode

    mutable struct BuilderState
        vertices::Matrix{S}
        indices::Vector{I}
        actuators::Vector{Tuple{I,I}}
        mode::PlacementMode
        mode_state::Any
    end

    function create()
        state = Observable(BuilderState(Matrix{S}(undef, 2, 0), Vector{I}(), Vector{Tuple{I,I}}(), SelectMode, ()))

        fig = Figure()

        ax = Axis(
            fig[1, 1],
            aspect = 1,
            limits = (-1.0, 1.0, -0.1, 1.9),
            title = lift(state) do st
                if st.mode == SelectMode
                    "- Select (S) - | Vertex (V) | Triangle (T) | Actuator (A)"
                elseif st.mode == VertexMode
                    "Select(S) | - Vertex (V) - | Triangle (T) | Actuator (A)"
                elseif st.mode == TriangleMode
                    "Select(S) | Vertex (V) | - Triangle (T) - | Actuator (A)"
                elseif st.mode == ActuatorMode
                    "Select(S) | Vertex (V) | Triangle (T) | - Actuator (A) -"
                end
            end
        )

        deactivate_interaction!(ax, :rectanglezoom)

        mesh!(fig[1, 1],
            lift(state) do st
                [ones(2, 3) * NaN st.vertices]
            end,
            lift(state) do st
                [[1, 2, 3]; st.indices[1:(length(st.indices)÷3)*3] .+ 3]
            end
        )

        scatter!(
            fig[1, 1],
            lift(state) do st
                st.vertices
            end
        )

        mesh!(
            fig[1, 1],
            lift(state) do st
                if length(st.actuators) < 1
                    ones(2, 3) * NaN
                else
                    vertices = Matrix{S}(undef, 2, 0)

                    for (i, j) in st.actuators
                        vi = st.vertices[:, i]
                        vj = st.vertices[:, j]

                        dir = normalize(vj - vi)

                        delta = [-dir[2], dir[1]] * 0.01

                        vertices = [vertices vi - delta vi + delta vj - delta vj + delta]
                    end

                    vertices
                end

            end,
            lift(state) do st
                if length(st.actuators) < 1
                    ones(Int64, 3)
                else
                    indices = Vector{I}()

                    for i = 1:length(st.actuators)
                        append!(indices, [0, -1, -2, 0, -2, -3] .+ 4 * i)
                    end

                    indices
                end
            end
        )

        scatter!(lift(state) do st
            if st.mode == SelectMode && st.mode_state isa I
                i = convert(I, st.mode_state)
                Point2f(st.vertices[:, i])
            else
                Point2f(NaN, NaN)
            end
        end)

        on(events(fig).keyboardbutton) do event
            st = state.val

            on_kb(st, event)
        end

        on(events(fig).mousebutton) do event
            on_click(state.val, event, Vector(mouseposition(ax.scene)))
        end

        state, fig
    end

    function on_mode_start(st::BuilderState)
        if st.mode == TriangleMode
            st.mode_state = Vector{I}()
        elseif st.mode == ActuatorMode
            st.mode_state = Vector{I}()
        end
    end

    function on_kb(st::BuilderState, event)
        if event.key == Keyboard.v
            switch_mode(st, VertexMode)
        elseif event.key == Keyboard.a
            switch_mode(st, ActuatorMode)
        elseif event.key == Keyboard.t
            switch_mode(st, TriangleMode)
        elseif event.key == Keyboard.s
            switch_mode(st, SelectMode)
        elseif event.key == Keyboard.w && event.action == Keyboard.press
            save(st)
        end

        if st.mode == SelectMode
            if event.key == Keyboard.delete && event.action == Keyboard.press && st.mode_state isa I
                i = convert(I, st.mode_state)

                st.vertices[:, i] = [NaN, NaN]

                indices = []

                for k = 1:length(st.indices)÷3
                    tri = 3k-2:3k
                    if !any(st.indices[tri] .== i)
                        append!(indices, st.indices[tri])
                    end
                end

                actuators = []

                for (ai, aj) in st.actuators
                    if ai != i && aj != I
                        push!(actuators, (ai, aj))
                    end
                end

                st.indices = indices
                st.actuators = actuators

                st.mode_state = ()
            end
        end
    end

    function on_click(st::BuilderState, event, mpos)
        if st.mode == SelectMode
            if event.action == Mouse.press && event.button == Mouse.left

                st.mode_state = ()

                for i = 1:size(st.vertices)[2]
                    if norm(st.vertices[:, i] - mpos) < 0.05
                        st.mode_state = i
                        break
                    end
                end
            end
        elseif st.mode == VertexMode
            if event.action == Mouse.press && event.button == Mouse.left
                add_vertex!(st, mpos)
            end
        elseif st.mode == TriangleMode
            if event.action == Mouse.press && event.button == Mouse.left
                vertex_found = false

                tri = convert(Vector{I}, st.mode_state)

                for i = 1:size(st.vertices)[2]
                    if norm(st.vertices[:, i] - mpos) < 0.05
                        push!(tri, i)
                        vertex_found = true
                        break
                    end
                end

                if !vertex_found
                    add_vertex!(st, mpos)
                    push!(tri, size(st.vertices)[2])
                end

                if length(tri) == 3
                    add_indices!(st, tri)
                    switch_mode(st, TriangleMode)
                end
            end
        elseif st.mode == ActuatorMode
            if event.action == Mouse.press && event.button == Mouse.left
                act = convert(Vector{I}, st.mode_state)

                for i = 1:size(st.vertices)[2]
                    if norm(st.vertices[:, i] - mpos) < 0.05 && (length(act) == 0 || act[1] != i)
                        push!(act, i)
                        break
                    end
                end

                if length(act) == 2
                    push!(st.actuators, (act[1], act[2]))
                    switch_mode(st, ActuatorMode)
                end
            end
        end
    end

    function switch_mode(st::BuilderState, mode::PlacementMode)
        on_mode_end(st)
        st.mode = mode
        on_mode_start(st)
    end

    function on_mode_end(st::BuilderState) end

    function redraw(state::Observable{BuilderState})
        notify(state)
    end

    function add_vertex!(state::BuilderState, vertex::Vector)
        state.vertices = [state.vertices vertex]
    end

    function add_indices!(state::BuilderState, indices::Vector{I})
        for index in indices
            push!(state.indices, index)
        end
    end

    function save(st::BuilderState)

        filename = save_dialog("Save...", GtkNullContainer(), (GtkFileFilter("*.json", name = "All supported formats"), "*.json"))

        if filename != ""

            n = size(st.vertices)[2]

            vertices = []

            vertmap = zeros(I, n)

            vi = 1
            for i = 1:n
                if !any(isnan.(st.vertices[:, i]))
                    append!(vertices, st.vertices[:, i])

                    vertmap[i] = vi

                    vi += 1
                end
            end

            indices = [vertmap[i] for i in st.indices]
            actuators = [(vertmap[i], vertmap[j]) for (i, j) in st.actuators]

            open(filename, "w") do io
                write(io, JSON.json(Dict("vertices" => vertices, "indices" => indices, "actuators" => actuators)))
            end
        end
    end
end

function test()
    builder, fig = SimulationBuilder.create()
    display(fig)


    while true
        sleep(0.1)
        SimulationBuilder.redraw(builder)
    end

end

test()