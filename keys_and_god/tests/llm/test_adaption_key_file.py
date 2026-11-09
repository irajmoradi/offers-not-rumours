"""load_adaption_key: a key file may carry a label or other lines around the pt_live_ token (fake keys only)."""
from swarm.llm.secrets import load_adaption_key, redact


def test_labelled_key_file_extracts_token(tmp_path):
    f = tmp_path / "k"
    f.write_text("label: pt_live_FAKEabc_123-xyz\n")
    assert load_adaption_key(f, env={}).get_secret_value() == "pt_live_FAKEabc_123-xyz"
    assert "pt_live_FAKEabc_123-xyz" not in redact("token pt_live_FAKEabc_123-xyz here")


def test_multi_line_file_takes_last_token_like_lfm_loaders(tmp_path):
    f = tmp_path / "k"
    f.write_text("# Noor's key\nold = pt_live_FAKEold\nkey=pt_live_FAKEnew  # current\n")
    assert load_adaption_key(f, env={}).get_secret_value() == "pt_live_FAKEnew"


def test_bare_raw_key_file_unchanged(tmp_path):
    f = tmp_path / "k"
    f.write_text("  rawFAKEkey-without-prefix\n")
    assert load_adaption_key(f, env={}).get_secret_value() == "rawFAKEkey-without-prefix"
    g = tmp_path / "k2"
    g.write_text("pt_live_FAKEbare\n")
    assert load_adaption_key(g, env={}).get_secret_value() == "pt_live_FAKEbare"


def test_missing_file_falls_back_to_env(tmp_path):
    assert load_adaption_key(tmp_path / "none", env={"ADAPTION_API_KEY": "pt_live_FAKEenv"}).get_secret_value() \
        == "pt_live_FAKEenv"
    assert load_adaption_key(tmp_path / "none", env={}) is None
