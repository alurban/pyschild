# Contributing to PySchild

## Reporting Issues

When opening an issue to report a problem, please try to provide a minimal code
example that reproduces the issue along with details of the operating
system and the Python, NumPy, Astropy, and PySchild versions you are using.

## Contributing Code

**Imposter syndrome disclaimer**: We want your help and you are welcome here.

Most of us, especially early in our careers, have an inner monologue justifying
to ourselves that we're not ready to be an open source contributor. That our
skills aren't nearly good enough to meaningfully help, or that we wouldn't even
know where to start. What could I *possibly* offer a project like this one?

We want to assure you that this couldn't be further from the truth: if you can
write code at all, you can contribute code to open source. It is not only a
fantastic way to advance one's own coding skills, but also a pragmatic way
to learn how to do collaborative science, on small scales and large. Not a
single one of us was born writing perfect code, after all, and even veteran
professionals periodically learn new languages (or risk falling behind).

**Writing perfect code is never the measure of a good developer.** Rather, it's
imperative to try to create something, make mistakes, and learn from those
mistakes and from openly sharing with others. That's how we all improve, and
as a community we are happy to help others learn.

Being an open source contributor also doesn't just mean writing code. You can
help out enormously by writing documentation, tests, or even giving feedback
about the project, which very much includes feedback about the contribution
process. This kind of input may even be the most valuable to the project in the
long term, because you're coming into it with fresh eyes, so you can more
easily see the errors and assumptions that seasoned contributors have glossed
over.

**Note:** This disclaimer was originally written by
[Adrienne Lowe](https://github.com/adriennefriend) for a
[PyCon talk](https://www.youtube.com/watch?v=6Uj746j9Heo), and was adapted for
PySchild based on its use in the [Astropy](https://github.com/astropy/astropy/)
and [GWpy](https://github.com/gwpy/gwpy/) contributing guides.

## Development model

This repository uses the [GitHub flow](https://guides.github.com/introduction/flow/)
collaborative development model.

In short, contributions to PySchild are made via pull requests from GitHub
users' forks of the main [pyschild repository](https://github.com/pyschild/pyschild).
The basic idea is to use the `master` branch of your fork as a way of keeping
your fork up-to-date with other contributors' changes that have been merged
into the main repo, then adding/changing new features on a dedicated *feature
branch* for each development project.

If this is your first contribution, make sure you have a GitHub account
(signing up is free of charge) and set up a development sandbox as follows:

*   Create the fork (if needed) by clicking *Fork* in the upper-right corner of
    <https://github.com/pyschild/pyschild/>. (This only needs to be done once,
    ever.)

*   From the command-line, if you haven't already, clone your fork (replace
    `<username>` with your GitHub username):

    ```bash
    git clone https://github.com/<username>/pyschild.git pyschild-fork
    cd pyschild-fork
    ```
  
*   Link your cloned fork to the upstream "main" repo:

    ```bash
    git remote add upstream https://github.com/pyschild/pyschild.git
    ```

For each development project:

*   Pull changes from the upstream "main" repo onto your fork's `master` branch
    to pick up other people's changes, then push to your remote to update your
    fork on github.com

    ```bash
    git pull --rebase upstream master
    git push
    ```

*   Create a new branch for this project with a short, descriptive name

    ```bash
    git checkout -b my-project
    ```

*   Make commits to this branch

*   Push changes to your remote on github.com

    ```bash
    git push -u origin my-project
    ```

*   Open a *merge request* (also known as a *pull request*) on github.com and
    tag the lead developer (@alurban) to initiate an interactive code review

*   When the request is merged, you should "delete the source branch" (there's a
    button) to keep your fork clean, and delete it from your local clone:

    ```bash
    git checkout master
    git branch -D my-project
    git pull upstream master
    ```

That's all there is to it!

**Note:** merge requests should be as simple as possible to keep the repository
clean, so if you have multiple contributions in mind, they should be split up
over multiple merge requests.

### Testing

All code contributions should be accompanied by unit tests to be executed with
[`pytest`](https://docs.pytest.org/en/latest/), and should cover all new or
modified lines as measured by [CodeCov](https://codecov.io).

On the command-line, you can run the test suite from the root of your cloned
repository via:

```bash
python -m pytest pyschild/
```
