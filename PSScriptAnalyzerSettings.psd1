# PSScriptAnalyzer configuration for scripts/.
#
# This file exists so the PSAvoidUsingWriteHost exemption is one reviewable decision rather
# than a [SuppressMessageAttribute] pasted into every operator-facing script. Inline
# suppressions are banned repo-wide (AGENTS.md, scripts/ci/check-inline-ignores.py), and
# PSScriptAnalyzer attributes are suppressions like any other -- the checker simply has no
# pattern for them yet, which made them invisible rather than allowed.
#
# Why the rule does not apply here: every .ps1 under scripts/ is an operator-facing console
# tool whose coloured, immediate output IS its interface. Write-Output would put the report
# on the success stream, where a caller capturing the result would collect the prose along
# with the value it actually wanted -- which is the bug the rule normally prevents, inverted.
#
# Nothing else is excluded. Every other default rule runs, at all three severity levels --
# Error, Warning AND Information. Information is deliberately included rather than dropped:
# several rules that matter for these scripts (PSUseUsingScopeModifierInNewRunspaces among
# them) report at that level, and the gate is only useful if it fails on them. The same
# three levels are what tests/run_suite.sh applies, because it passes this file as -Settings.
@{
    IncludeDefaultRules = $true
    Severity            = @('Warning', 'Error', 'Information')
    ExcludeRules        = @('PSAvoidUsingWriteHost')
}
