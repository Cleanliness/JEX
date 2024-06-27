
    # a better correctness test
    # qk_rope_fn = partial(qk_rope, rd=gem.rope)
    # all_acts = []
    # for i in range(18):
    #     test_x = jnp.ones((1, 1, 2048))
    #     test_res = apply_gemma_block(test_x, gem.blocks[i], mid_fn=qk_rope_fn)
    #     diff = test_res - correct_hiddens[i]
    #     all_acts.append(diff)

    #     plt.hist(diff.flatten(), bins=100)
    #     plt.savefig(f"layer_{i}_diff.png")
    #     plt.close()
    #     print("finished layer", i)

    # all_acts = jnp.array(all_acts)
    # plt.hist(all_acts.flatten(), bins=500)
    # plt.savefig("all_layer_diff.png")