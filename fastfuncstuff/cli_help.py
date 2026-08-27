"""Help formatting shared by every ``ffs_*`` command-line tool.

Deliberately torch-free and dependency-free.  ``cli_utils`` imports torch at
module scope, and several CLIs defer that import to keep ``-help`` fast (the
startup path is ~1.9 s and most of what remains is torch); a help formatter has
no business dragging it in.  The completion generator imports from here too, so
what ``-help`` prints and what TAB offers are computed from one function.
"""

from __future__ import annotations

import argparse
import re

__all__ = [
    "FfsArgumentParser",
    "FfsHelpFormatter",
    "ScannableHelpFormatter",
    "canonical_option_strings",
    "spelling_variants",
    "suggest",
]


def canonical_option_strings(option_strings: list[str]) -> list[str]:
    """Drop pure spelling variants, keep genuinely different names.

    Every ffs flag accepts both ``-foo-bar`` and ``-foo_bar``, and a few also
    carry a dropped-separator form (``-no_coverage`` / ``-nocoverage``).  None
    of those is a new thing the user can do, so they collapse onto the first
    spelling argparse was given -- which is the documented one.

    The key strips ``-`` and ``_`` entirely rather than normalising ``_`` to
    ``-``: normalising alone misses ``-nocoverage`` vs ``-no_coverage``, which
    is how eleven of those slipped through into the generated completions.

    Anchoring on the FIRST spelling rather than "prefer the dash form" matters:
    flags like ``-drop_first`` are documented with the underscore and only
    alias to ``-drop-first``, so a blanket dash rule hides the primary name and
    surfaces the alias instead.

    Shared by the help formatter and the completion generator so that what
    ``-help`` prints and what TAB offers cannot drift apart.
    """
    seen: set[str] = set()
    out: list[str] = []
    for opt in option_strings:
        key = opt.replace("_", "").replace("-", "")
        if key in seen:
            continue
        seen.add(key)
        out.append(opt)
    return out


class FfsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """The one help formatter for every ``ffs_*`` tool.

    Four behaviours, each of which was invented independently in a different
    CLI before being collected here:

    * **Honour newlines in help.** argparse re-flows every help string into one
      paragraph, which turns a flag documenting four choices into an
      unscannable wall.  Explicit newlines survive, and a wrapped continuation
      carries a hanging indent so a choice stays visually one block.
    * **Keep the epilog's layout** (from ``RawDescriptionHelpFormatter``), so
      hand-laid example blocks are not reflowed.
    * **Show each flag's default** -- except ``None`` and ``False``, which mean
      "computed after parsing" and "off"; the help text already says so, and
      they were a third of the ``(default: ...)`` lines.
    * **Collapse spelling-variant aliases** to one entry.  A real alternative
      name (``-max_pcs``, ``-parametrisation``) still prints, because it
      carries information; ``-foo-bar`` alongside ``-foo_bar`` does not.

    Written for ffs_locomoco and ffs_fitbasis; shared so the family's help
    reads the same way and matches the shell completions flag for flag.
    """

    def _get_help_string(self, action):
        if action.default is None or action.default is False:
            # "(default: None)" is never information -- the real default is computed
            # after parsing, or the flag is simply optional.  False is "off", which
            # is what the flag's own help already says.
            return action.help
        if isinstance(action, argparse._StoreFalseAction):
            # A -no_foo switch shares its dest with the positive partner, so the
            # default argparse would print is the PARTNER's value, not this flag's.
            return action.help
        if "default" in (action.help or "").lower():
            # The author already wrote the default into the sentence; appending a
            # second one is how "(default: 20). (default: 20)" happens.
            return action.help
        text = super()._get_help_string(action)
        if text and "\n" in text and text.endswith(")"):
            # " (default: X)" tacked onto the last line of a choice list reads as part
            # of that choice. Break it out onto its own line once the help is multi-line.
            head, _, tail = text.rpartition(" (default:")
            return f"{head}\n(default:{tail}" if head else text
        return text

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        shown = canonical_option_strings(action.option_strings)
        if action.nargs == 0:
            return ", ".join(shown)
        args = self._format_args(action, self._get_default_metavar_for_optional(action))
        return ", ".join(shown) + " " + args

    def _split_lines(self, text: str, width: int) -> list[str]:
        import textwrap

        out: list[str] = []
        for line in text.splitlines():
            if not line.strip():
                out.append("")
                continue
            # An indented "  key   meaning" line hangs under the MEANING, so a wrapped
            # choice stays visually one block instead of drifting back under its key.
            match = re.match(r"(\s+\S+\s\s+)", line)
            lead = len(line) - len(line.lstrip())
            # Cap the hang: a long key (CONDITION-PAIRED) would otherwise push its own
            # continuations most of the way across the terminal.
            hang = " " * min(len(match.group(1)), lead + 12) if match else " " * lead
            out.extend(textwrap.wrap(line, width, subsequent_indent=hang) or [""])
        return out


# The name this class shipped under while it lived only in the GLM tools.
ScannableHelpFormatter = FfsHelpFormatter


def spelling_variants(option_string: str) -> list[str]:
    """The other ways to write one flag name, if any.

    ``-event_ignore`` and ``-event-ignore`` are the same request, and which one
    a tool happens to have registered is an accident of who wrote it: 646 of
    the 1113 multi-word flags in the toolbox accept only one of the two, so
    ``-event_ignore`` works in ffs_denoise and fails in ffs_deconvolve.  A
    convention honoured 42% of the time is not a convention.

    Only the separator moves.  A flag whose name has no separator at all has no
    variant, and a name mixing both (``-hrf-n_shapes``) yields the all-hyphen
    and all-underscore forms.
    """
    body = option_string.lstrip("-")
    if "_" not in body and "-" not in body:
        return []
    dashes = "-" * (len(option_string) - len(body))
    forms = {dashes + body.replace("_", "-"), dashes + body.replace("-", "_")}
    return sorted(forms - {option_string})


def suggest(action: argparse.Action, values) -> argparse.Action:
    """Offer ``values`` at the shell, without making them the only legal ones.

    ``choices=`` is the right tool when the set really is closed, and 235 flags
    use it.  This is for the ones where it is not: ``-device`` also takes
    ``cuda,0``, ``-polort`` also takes ``A``, ``-adjust_dof`` takes a map OR a
    number.  Those flags used to complete nothing at all -- or, in the single
    case anybody had bothered with, be special-cased by matching on the string
    "device" inside the completion generator.

    Returns the action, so it can wrap an ``add_argument`` call in place.
    """
    # setattr, matching the getattr on the reading side: argparse.Action has no
    # slot for this and no hook to add one.
    setattr(action, "ffs_suggest", [str(v) for v in values])  # noqa: B010
    return action


class FfsArgumentParser(argparse.ArgumentParser):
    """``ArgumentParser`` that accepts both spellings of every flag it is given.

    Registering ``-foo-bar`` also makes ``-foo_bar`` parse, and vice versa.
    The extra spelling is deliberately NOT added to ``action.option_strings``:
    it must not appear in ``-help``, in the usage line, or in the shell
    completions, all three of which read ``option_strings``.  It exists only to
    be typed.

    Aliases materialise at parse time rather than at ``add_argument`` time so
    that a parser being introspected -- by the completion generator, or by
    ``-help`` -- sees exactly the documented names.

    Abbreviation is repaired at the same time.  argparse counts option STRINGS,
    not flags, when deciding whether a prefix is ambiguous, so a toolbox that
    registers two spellings of everything makes ``-dro`` report

        ambiguous option: -dro could match -drop_first, -drop-first,
                          -drop_last, -drop-last

    -- four names for the two flags that are actually in conflict.  Deduping by
    action fixes the message, and hidden aliases stay out of prefix matching
    entirely, so adding 646 of them cannot create a new ambiguity.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", FfsHelpFormatter)
        super().__init__(*args, **kwargs)
        self._ffs_alias_strings: set[str] = set()
        self._ffs_aliases_applied = False

    def _ffs_apply_spelling_aliases(self) -> None:
        """Make every registered flag's other spelling parse. Idempotent."""
        if self._ffs_aliases_applied:
            return
        self._ffs_aliases_applied = True
        for action in self._actions:
            for option in action.option_strings:
                for variant in spelling_variants(option):
                    # Never shadow a real flag: if the variant is already taken,
                    # by this action or another, the author's choice wins.
                    if variant not in self._option_string_actions:
                        self._option_string_actions[variant] = action
                        self._ffs_alias_strings.add(variant)

    def _get_option_tuples(self, option_string):
        matches = super()._get_option_tuples(option_string)
        seen = set()
        out = []
        for tup in matches:
            action, matched = tup[0], tup[1]
            if matched in self._ffs_alias_strings:
                continue  # an undocumented spelling never wins an abbreviation
            if id(action) in seen:
                continue  # one flag, however many spellings: not an ambiguity
            seen.add(id(action))
            out.append(tup)
        return out

    def parse_known_args(self, args=None, namespace=None):
        self._ffs_apply_spelling_aliases()
        return super().parse_known_args(args, namespace)
