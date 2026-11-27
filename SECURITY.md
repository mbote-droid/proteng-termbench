# Security

## Reporting

Please do not open a public issue for a security problem. Use GitHub's private
vulnerability reporting for this repository, or email mbotesamuel9@gmail.com.
I will reply within a week.

## Scope

The graders run submitted code. Untrusted submissions should be graded with
`--sandbox-user` (or in the Harbor containers), never with `--trusted`;
`docs/DESIGN.md` describes the isolation. Reports about escaping that
isolation, reading hidden seeds or reference answers from a submission, or
tampering with a verdict are in scope. The agent harness in `agent_eval/` is a
development tool and does not disable the network.
